#!/usr/bin/perl -w
require LWP::UserAgent;

$num_args = $#ARGV + 1;
if ($num_args != 2) {
    print "\nUsage: verify_hostname_to_cert.pl <url> <port>\n";
    exit;
}

$hostname=$ARGV[0];
$port=$ARGV[1];

my $ua = LWP::UserAgent->new;
$ua->ssl_opts( verify_hostname => 1 );
my $response = $ua->get("$hostname:$port");
print $response->is_success ? 'OK' : 'INVALID', "\n";
