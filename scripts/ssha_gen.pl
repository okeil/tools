#! /usr/bin/perl
use Digest::SHA1;
use MIME::Base64;
$ctx = Digest::SHA1->new;
$ctx->add('pwd');
$ctx->add('test');
$hashedPasswd = '{SSHA}' . encode_base64($ctx->digest . 'test' ,'');
print 'userPassword: ' .  $hashedPasswd . "\n";
