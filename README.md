# gitlab_cicd_gpu_runner_template

This is a template for a gitlab ci/cd pipeline for training on a GPU or in general using the gitlab cicd runner pipeline.

As requirement, you need a gitlab account and access to the file server to work with this setup. 

Large files should not be stored in the gitlab repository, but in the file server, and be copyied from the fileserver
in the docker container when running the script. See more information in the gitlab-ci.yml

## File Server

You can access the file server from within the IRP network. Please contact Erik Reimer if you have questions regarding VPN access to the IRP network.
In Ubuntu use FileManagement to connect to server: "smb://fs.rob.cs.tu-bs.de/students/" and log in with your credentials.
Please only use your directory "j_mohr". The gitlab runner can only access the public directory ! 

## Usage

Put all your code in the project repository. 
The code executed on the gitlab runner is in the gitlab-ci.yml file.
Please read it carefully and change it to your needs.
But leave this project as template and keep it as it is! Feel free to copy for your own projects.

IMPORTANT: If you have secret keys, they can be stored save via the gitlab web interface, under Settings -> CI/CD -> Variables. 
They can be masked for secure reasons! Not put any secret information inside the repository!

For more information about gitlab cicd, take a look at official documentation: https://docs.gitlab.com/ci/


## Important notes:

Print statements are often not directly printed in the cicd pipeline. You can use sys.flush() in your scripts to force printing.