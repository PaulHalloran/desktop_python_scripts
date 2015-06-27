'''
1st install paramiko throiugh package manager
'''

import paramiko
import iris


ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

server_address='geog-mce.ex.ac.uk'
usr_name='ph290'
psswrd='whetever_it_is'

ssh.connect(server_address, username=usr_name, password=psswrd)

loc_of_data='/home/ebs/ph290/data/'
example_file='HadISST_sst.nc'

ftp = ssh.open_sftp()
sftp.get(loc_of_data+example_file, example_file)
#but this copies it across to current location
#x=iris.load_cube(example_file)

#would be better if we coudl do something like this:
#file=ftp.file(loc_of_data+example_file, "rb", -1)
#x=iris.load_cube(file)
#but does not seem to work

ftp.close()
ssh.close()
