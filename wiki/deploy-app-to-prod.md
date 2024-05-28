## Steps to deploy production neuvue-app in AWS
(WIP)

System Requirements:
* Amazon Web Services (AWS) account (TODO: add estimate for monthly cost)
* Google Cloud Platform (GCP) account (just for OAuth, no billed services will be used)
* Git
* Miniconda
* EB CLI (instructions to install are below)
* MySQL Workbench

### Set up new domain
* Open AWS console
* Buy and register a domain in Route 53 (e.g. neuvue.io)
* Create a certificate for the fully qualified domain name (e.g. neuvue.io or app.neuvue.io) in Amazon Certificate Manager with DNS validation
* DNS validate the certificate by clicking "Create records in Route 53"

### Create new OAuth 2.0 Client ID
* Open GCP console. We will set up Google authentication here so that users can use a Gmail account to log in.
* Navigate to APIs & Services > Credentials > Create Credentials > OAuth Client ID
  * Use fully qualified domain name (FQDN) for authorized JS origin and <FQDN>/accounts/google/login/callback/ as authorized redirect URI
* Save client ID and client secret for later 

### Install EB CLI
* Instructions here https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install.html 
* Also make sure to add credentials to a profile in local path `~/.aws/config`, instructions here https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html 

### Create new Elastic Beanstalk environment
* Return to AWS console. Navigate to the Elastic Beanstalk homepage. We will deploy the website on Elastic Beanstalk.
* Click `Create environment`. Most of the defaults are fine, can reference existing Boss deploy with questions or ask Danny. Make sure to choose Python 3.9 or lower, no VPC subnet, and load balancer.
* After environment is initialized, add the following env variables:
  * CAVECLIENT_TOKEN: see existing deploy
  * DJANGO_SECRET_KEY: can be created with `python3 -c 'import secrets; print(secrets.token_hex(100))'`
  * DJANGO_SETTINGS_MODULE: neuvue.settings
  * NEUVUEQUEUE_ACCESS_TOKEN: see existing deploy
  * NEUVUEQUEUE_REFRESH_TOKEN: see existing deploy
  * PYTHONPATH: /var/app/current:$PYTHONPATH
* Also replace the WSGIPath configuration with `neuvue.wsgi:application`

### Set up local NeuVue to deploy from
* Locally, navigate to the directory you'd like to work from and run `git clone https://github.com/aplbrain/neuvue-app`
* `cd neuvue-app/neuvue_project`
* `conda create --name neuvue python=3.9`
* `conda activate neuvue`
* `pip install -r requirements.txt`
* In `.ebextensions/https.config` change the cert to the one you created earlier during [Set up new domain](#set-up-new-domain)
* Run `eb init` to connect your local app to the environment you made in the console. Use `--profile` if you've defined multiple profiles in `~/.aws/config`
* Add domain to list of allowed hosts in `settings.py`

### Create RDS Database
* Return to the AWS Console and navigate to the RDS homepage. The last thing to configure is a very small relational database for the Django admin console.
* Click Create Database. Once again, most of the defaults are fine. The database will be very small. Reference the existing Boss deploy with questions or ask Danny. Be careful not to choose a vpc subnet, to open db to all internet traffic, and to set a memorable password
* Get off the VPN if you are on it
* Open MySQL Workbench and connect to the new RDS instance. To connect you will need the RDS hostname and port (found under Endpoint & Port on the RDS instance's console page) and the username and password you created during the instance setup. If you forgot the username and password, they can be found under Configuration > Master username/Master password.
* Add a new database inside the RDS instance once you are connected by executing the command `CREATE DATABASE <db-name>;` ebdb is a good name
* Set temporary local environment variables to point your local NeuVue clone to the new database
  * RDS_DB_NAME: whatever you just named it, e.g. ebdb
  * RDS_HOSTNAME: same as MySQL Workbench setting
  * RDS_PORT: same as MySQL Workbench setting
  * RDS_USERNAME: same as MySQL Workbench setting
  * RDS_PASSWORD: same as MySQL Workbench setting
* Add all the table schemas by running `python manage.py migrate`
* Set same the five environment variables in Elastic Beanstalk. The relevant page is Elastic Beanstalk environment home page > Configuration > Updates, monitoring, and logging > Edit > Scroll to the bottom of the page > Environment properties
* Go back to MySQL Workbench and add a row to each of three tables
  * `socialaccount_socialapp`: oauth client id and secret columns come from [Create new OAuth 2.0 Client ID](#create-new-oauth-2.0-client-id)
  * `django_site`: FQDN in two columns: domain and name
  * `socialaccount_socialapp_sites`: connect social app to django site with the table id numbers from the two entries just created
* Add `SITE_ID=<table ID for django_site>` to settings.py locally
* `eb deploy` locally
* If site related errors persist do an entire environment rebuild to ensure that db cache is not causing errors
* Create account to access admin dashboard with local `python manage.py createsuperuser`, make sure same environment variables are still set
  
### Other Admin Console things I had to configure
* Log in as myself to create an account for myself
* Give myself admin and super user privileges via the admin console while logged in as the superuser account created during [Create RDS Database](#create-rds-database)
* Create group AuthorizedUsers and add myself to it
* Add button groups before adding a new namespace - I think some of these should ship with NeuVue. We need to decide which ones
  
### Todos
* Need to fix redirect url in `settings.py` for token; is currently hardcoded