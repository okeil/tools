#!/usr/bin/python
import boto3
import json

client = boto3.client('route53')

external_zones = client.list_hosted_zones(
    MaxItems='100'
)

for zone in external_zones['HostedZones']:
    resource_records = client.list_resource_record_sets(
        HostedZoneId=zone['Id'][12:]
    )
    for rr in resource_records['ResourceRecordSets']:
        if rr['Type'] == 'A' or rr['Type'] == 'AAAA' or rr['Type'] == 'CNAME':
            print(rr['Name'])
            #print(rr['Name'] + " - " + rr['Type'])
        #if 'ResourceRecords' in rr:
        #    for val in rr['ResourceRecords']:
        #        print(val['Value'])
