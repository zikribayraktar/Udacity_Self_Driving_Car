

[//]: # (Image References)

[image1]: ./images/GCP_01.jpg
[image2]: ./images/GCP_02.jpg
[image3]: ./images/GCP_03.jpg
[image4]: ./images/GCP_04.jpg

## Setting up Google Cloud Platform Virtual Environment

The goals of this project are:
* Google Cloud Platform offers attractive features and good pricing, where I am aiming to duplicate the work I do on my PC as well run the deep learning projects instead of AWS.
* Explore and exploit any useful cloud applications and tools for self driving car projects.
* Install and run two virtual machines, one cpu-only vm, and one gpu vm.
   
---
We start with signing up for [Google Cloup Platform (GCP) Free Tier](https://cloud.google.com/free/). Follow the steps and create an
account so that Google can give you $300 credit and a year to spend it on GCP.

Your GCP platform console should look like this:
![GCP_Console][image1]

<br>
Once we are on the GCP console, we need to set up a virtual machine (VM). I'll show cpu-only VM first.
Find the menu on the left side of the console. Click on "Compute Engine" and then "VM Instances".
Click on "Create Instance" and select the specs of your VM. We need to give a name to the VM and select
the region. For "Machine Type", I selected 2 vCPUs, and for "Disk", I selected 10 GB with Ubuntu 16.04 LTS.
<br>
It will take a minute for GCP to initialize your VM and then you are good to go. It will look like this:
![GCP_VM_Instance][image2]

<br>
I blocked out the IPs. Note that External IP shown for your active VM is what we will use to connect to.
Also note the green check mark on the left hand side of your VM. It shows your VM is active and incurring cost.
<b>Make sure to stop your VM once you are done using it. Otherwise, it will continue running and GCP will charge you.</b>
To stop the VM, simply click on the three dots menu next to the VM, choose Stop option as seen above. Your VM will be still
on the GCP, but you will not be charged for it. When you need it, come back and Start your VM. Note that, each time we
stop and start a VM, the External IP maybe different. Make sure to connect to the right IP in your VNC.

<br><br>
Ok, we have a VM running, now click on the "SSH" box next to VM, and select "Open in browser window".
This will open a command window to your VM. I will take a shortcut here and give you a link to a nice tutorial for
[GUI for GCP VM Instances](https://medium.com/google-cloud/graphical-user-interface-gui-for-google-compute-engine-instance-78fccda09e5c)
These instructions are for Ubuntu 14.04 LTS but works just fine for our version Ubuntu 16.04 LTS as well. There are couple of things I did
differently. For VNC settings, I used the info [here](https://askubuntu.com/questions/800302/vncserver-grey-screen-ubuntu-16-04-lts).
When I setup my VM, I was on MIT campus and it turned out that 5901 port was blocked, as well as some others on the network. So, make sure that
your VNC is through an open port.

<br><br>
Let's start the VNC, enter the External IP address of your VM and port number. The VM throug my VNC looks like this:

![VNC_viewer][image3]

<br><br>
Let's install Google Chrome on our VM. Here are the [steps](https://askubuntu.com/questions/510056/how-to-install-google-chrome/510063) to
install it on your command line. Once, installed, type google-chrome on your VM command line and it will start running Chrome in GUI. We need this
for the Jupyter Notebook for our SDC projects.  Now, we can follow the instructions provided by the nice people of Udacity SDC program [here](https://github.com/udacity/CarND-Term1-Starter-Kit)
to install our Python environment.  I like Miniconda, so I followed those steps from the GitHub instructions, then imported the provided 
environment.yml.  Once the environment is ready, type 'source activate carnd-term1' and all libraries needed will be available to you for SDC term 1.

<br> Our final setup should look like this:
![JupyterNotebook][image4]

<br> Now, we practically duplicated our PC environment on GCP. Thanks to the good people of Udacity and kind people of the Internet, we
can set this up pretty easily as you have seen above. Happy coding! 
<br><br>

---
<b>DISCLAIMER:  Any information listed on these pages are educational purposes only. I do not guarantee it is bug free or it will even work for
you. Do not use in production environment nor in safety related environments. Use at your own risk!</b>