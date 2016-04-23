#!/usr/bin/python
import boto.emr
import boto
#from boto.emr.step import InstallHiveStep, HiveStep, ScriptRunnerStep
from datetime import datetime
from dateutil.relativedelta import relativedelta
from calendar import timegm
from urlparse import urlparse
from optparse import OptionParser
import time
import sys



parser = OptionParser()
parser.add_option("-m", "--month", dest = "month", type = "int")
parser.add_option("-y", "--year", dest = "year", type = "int")
parser.add_option("--domain-depth", dest = "domain_depth", type = "int", default = 2, help = "How many labels (root = 1) are to be processed for the domain")
parser.add_option("--num-nodes", dest = "num_nodes", type = "int", default = 51, help = "Number of AWS to execute on")
parser.add_option("--region", dest = "region", type = "string", default = "us-west-2", help = "AWS region")
parser.add_option("--master-type", dest = "master_type", type = "string", default = "g1.xlarge", help = "AWS master instance type")
parser.add_option("--worker-type", dest = "worker_type", type = "string", default = "m1.xlarge", help = "AWS worker instance type")
parser.add_option("--force", dest = "force", action = "store_true", default = False, help = "Force execution")
parser.add_option("--iam-role", dest = "iam_role", type = "string", default = "dns_analysis", help = "AWS IAM role")
(options, args) = parser.parse_args(sys.argv)
if not options.month:
  parser.error("month needs to be specified")
if not options.year:
  parser.error("year needs to be specified")



split_size = 128
#libs = [ "s3n://ariservices-runreports-pcap/ari-reports-0.0.1-SNAPSHOT.jar",
#         "s3n://ariservices-runreports-pcap/hadoop-pcap-serde-1.2-SNAPSHOT-jar-with-dependencies.jar" ]
input_location = "s3n://ariservices-pcap"
output_location = "s3n://ariservices-reports-pcap"
run_location = "s3n://ariservices-runreports-pcap/runs"
nodes = [ "mel", "syd", "per", "bne", "akl", "hkg", "tky", "jbo", "nbo", "ist", "mad", "ams", "sto", "bog", "sao", "mia", "tor", "sjc" ]
identifier = datetime.utcnow().strftime("%Y%m%d-%H%M")
logs_location = "s3n://ariservices-runreports-pcap/logs"
work_location = run_location + "/" + identifier
report_location = output_location + "/" + identifier
heap_file = "s3n://ariservices-runreports-pcap/set-hive-heap.sh"
check_num_jobflows = 10



conn_emr = boto.emr.connect_to_region(options.region)
tags = dict(map(lambda (k, v): (str(k), str(v)), options.__dict__.iteritems()))
if not options.force:
  for job in conn_emr.describe_jobflows(states = [ "BOOTSTRAPPING", "STARTING", "TERMINATED", "COMPLETED", "WAITING", "FAILED", "SHUTTING_DOWN", "RUNNING" ])[:check_num_jobflows]:
    cluster = conn_emr.describe_cluster(job.jobflowid)
    cluster_tags = {}
    for tag in cluster.__dict__["tags"]:
      cluster_tags[str(tag.key)] = str(tag.value)
    if 0 == cmp(tags, cluster_tags):
      print "Seems that this job has run already (Jobflow ID: %s). You should check what happened to the previous job first. If you are sure you want to proceed, set --force." % job.jobflowid
      sys.exit(1)


date_now = datetime.strptime("%s-%s" % (options.year, options.month), "%Y-%m")
date_before = date_now - relativedelta(days=1)
date_after = date_now + relativedelta(months=1)

start = timegm(date_now.utctimetuple())
stop = timegm(date_after.utctimetuple()) - 1

conn_s3 = boto.connect_s3()
input_bucket = conn_s3.get_bucket(urlparse(input_location).netloc)

def generateAlterTableStatement(date, loc, node):
  path = loc + "/" + node + "/";
  if len(input_bucket.get_all_keys(prefix=path, max_keys=1)) > 0:
    return "ALTER TABLE packets ADD PARTITION (year = " + str(date.year) + ", month = " + str(date.month) + ", day = " + str(date.day) + ", node = '" + node + "') LOCATION '" + input_location + "/" + path + "';";
  return "-- No data in: " + path

steps = [ InstallHiveStep(), ScriptRunnerStep(name="Set Hive heap", step_args=[heap_file]) ]

query = []
for lib in libs:
  query.append("ADD JAR " + lib + ";")
query.append("CREATE TEMPORARY FUNCTION ExtractTLD AS 'com.ariservices.hive.HiveExtractTLD';")
query.append("SET net.ripe.hadoop.pcap.io.reader.class=net.ripe.hadoop.pcap.DnsPcapReader;")
query.append("SET hive.input.format=org.apache.hadoop.hive.ql.io.CombineHiveInputFormat;")
query.append("SET mapred.max.split.size=" + str(split_size * 1024 * 1024) + ";")
query.append("CREATE EXTERNAL TABLE packets (ts bigint, protocol string, dns_question string, dns_qr string)")
query.append("PARTITIONED BY (year int, month int, day int, node string)")
query.append("ROW FORMAT SERDE 'net.ripe.hadoop.pcap.serde.PcapDeserializer'")
query.append("STORED AS INPUTFORMAT 'net.ripe.hadoop.pcap.io.PcapInputFormat' OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat';")
for node in nodes:
  query.append(generateAlterTableStatement(date_before, date_before.strftime("%Y-%m/%d"), node))
  counter = start
  while counter < stop:
    date_counter = datetime.utcfromtimestamp(counter)
    query.append(generateAlterTableStatement(date_counter, date_counter.strftime("%Y-%m/%d"), node))
    counter += 86400
  query.append(generateAlterTableStatement(date_after, date_after.strftime("%Y-%m/%d"), node))
query.append("INSERT OVERWRITE DIRECTORY '" + report_location + "'")
query.append("SELECT LOWER(ExtractTLD(SPLIT(dns_question, ' ')[0], %d)) AS domain, protocol, COUNT(*)" % options.domain_depth)
query.append("FROM packets")
query.append("WHERE ts>=" + str(start) + " AND ts<=" + str(stop))
query.append("AND dns_question IS NOT NULL")
query.append("AND dns_qr='false'")
query.append("GROUP BY LOWER(ExtractTLD(SPLIT(dns_question, ' ')[0], %d)), protocol" % options.domain_depth)
query.append("ORDER BY domain, protocol ASC;")

work_url = urlparse(work_location)
work_bucket = conn_s3.get_bucket(work_url.netloc)
work_bucket.new_key(work_url.path[1:] + ".hql").set_contents_from_string("\n".join(query))
conn_s3.close()

steps.append(HiveStep(name="Queries per TLD",
                      hive_file=work_location + ".hql"))

job_id = None
try:
  job_id = conn_emr.run_jobflow(name="Queries per TLD " + identifier,
                                steps=steps,
                                num_instances=options.num_nodes,
                                log_uri=logs_location,
                                #ec2_keyname="wnagele",
                                hadoop_version="1.0.3",
                                master_instance_type=options.master_type,
                                slave_instance_type=options.worker_type,
                                job_flow_role=options.iam_role,
                                service_role=options.iam_role)
  conn_emr.add_tags(job_id, tags)

  while True:
    status = conn_emr.describe_jobflow(job_id).state
    print "%s: jobid=%s identifier=%s status %s" % (datetime.now(), job_id, identifier, status)
    if "FAILED" == status or "TERMINATED" == status or "COMPLETED" == status:
      break
    time.sleep(300)
except KeyboardInterrupt:
  if job_id:
    print "Terminating job: %s" % job_id
    conn_emr.terminate_jobflow(job_id)
finally:
  conn_emr.close()
