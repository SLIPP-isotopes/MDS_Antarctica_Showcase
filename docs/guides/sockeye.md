# Sockeye

Author: Shirley Zhang

Date: June 2023

> **Sections**
> 
> 1. Introduction
> 2. Logging in
> 3. Navigating and setting up the directories


## 1. Introduction

This document serves as a comprehensive guide to using UBC ARC's Sockeye high computing platform. It details the workflow the SLIPP team followed from May - June 2023 to set-up and train Gaussian Process models on Sockeye. The document can be followed whether you are starting from a completely new allocation (see sections marked "\[new allocation\]") or whether you are continuing the SLIPP project within the `st-aorsi-1` allocation (see sections marked "\[st-aorsi-1 allocation\]").

For a more simplified version of using Sockeye to run Gaussian Process models, see the following notebook: []. The Sockeye tutorial in this notebook assumes that you are part of the `st-aorsi-1` allocation and have access to the group's directories. 

**Overview of Workflow:**
- Apply to obtain resources from UBC ARC's Sockeye cloud computing platform. Professor Anais Orsi did this for us. Then, obtain permission to connect to Sockeye with your CWL. (Professor Anais Orsi will have to contact Sockeye and give them your cwl). 
- Then, do the first-time set-up stuff where you install Cisco AnyConnect and enhanced CWL (two-factor authentication).
- Next, try ssh-ing into Sockeye.
- Make a plan of the GP jobs you would like to submit. It would be helpful to create a spreadsheet.
- Next, make sure all the necessary scripts and data are set up in the directories. You can use the already set-up st-aorsi-1 group or set-up your own if you have a new allocation.
- Create an environment, or use the one already set-up for st-aorsi-1. 


> SLIPP's allocation code: `st-aorsi-1`

## 2. Logging in 

**First-time set-up**

1. Make sure your UBC cwl has permissions to access Sockeye ([Apply for Sockeye Standard](https://arc.ubc.ca/compute-storage/ubc-arc-sockeye/apply-sockeye-standard))
2. Set up UBC myVPN: https://it.ubc.ca/services/email-voice-internet/myvpn/setup-documents 
3. Set up UBC enhanced CWL (i.e. two-factor authentication, Duo Push) 

For more information on first-time set-up, see the [Quickstart Guide](https://confluence.it.ubc.ca/display/UARC/Quickstart+Guide).

**Logging in each time**

1. Open Cisco, connect to UBC VPN 
2. Open terminal, ssh into Sockeye (replace with your UBC cwl username): 
```
$ ssh cwl@sockeye.arc.ubc.ca
```
3. 2-factor authentication 

---

## 3. Navigating the directories 

#### **General overview** 

There are three main directories:

1. **Home:** `/home/cwl/`

> **Purpose:** Coding scripts, pipelines you don't want to share
> **Access:** Only available to you
> **Storage:** 50 GB + unlimited files

2. **Project:** `/arc/project/allocation-code/`

> **Purpose:** For raw inputs, project data, locally installed software, virtual environments shared with your team, and outputs you want to keep
> **Access:** Shared with everyone in your allocation/team (read-only)
> **Storage:** 5 TB + unlimited files

3. **Scratch:** `/scratch/allocation-code/`

> **Purpose:** For batch jobs, job scripts, and intermediate files 
> **Access:** Shared with everyone in your allocation/team (readable/writable)
> **Storage:** 5 TB + limit of 1,000,000 files

 
