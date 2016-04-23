import getpass, poplib
user = 'username' 
Mailbox = poplib.POP3('hostname/ip', '110') 
Mailbox.user(user) 
Mailbox.pass_('ss') 
numMessages = len(Mailbox.list()[1])
for i in range(numMessages):
    for msg in Mailbox.retr(i+1)[1]:
        print msg
Mailbox.quit()
