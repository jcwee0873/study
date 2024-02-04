---
title: Developer Environment
weight: 12030
indent: true
---

# Install Kubernetes

You can choose any Kubernetes install of your choice. The test framework only depends on `kubectl`
being configured. To install `kubectl`, please see the [official guide](https://kubernetes.io/docs/tasks/tools/#kubectl).

# Minikube

The developers of Rook are working on Minikube and thus it is the recommended way to quickly get
Rook up and running. Minikube should not be used for production but the Rook authors
consider it great for development. While other tools such as k3d/kind are great, users have faced
issues deploying Rook.

**Always use a virtual machine when testing Rook. Never use your host system where local devices may mistakenly be consumed.**

To install Minikube follow the [official
guide](https://minikube.sigs.k8s.io/docs/start/). It is recommended to use the
kvm2 driver when running on a Linux machine and the hyperkit driver when running on a MacOS. Both
allow to create and attach additional disks to the virtual machine. This is required for the Ceph
OSD to consume one drive.  We don't recommend any other drivers for Rook. You will need a Minikube
version 1.23 or higher.

Starting the cluster on Minikube is as simple as running:

```console
# On Linux
minikube start --disk-size=40g --extra-disks=1 --driver kvm2

# On MacOS
minikube start --disk-size=40g --extra-disks=1 --driver hyperkit
```

It is recommended to install a Docker client on your host system too. Depending on your operating
system follow the [official guide](https://docs.docker.com/engine/install/binaries/).

Stopping the cluster and destroying the Minikube virtual machine can be done with:

```console
minikube delete
```

## Install Helm

Use [helm.sh](/tests/scripts/helm.sh) to install Helm and set up Rook charts defined under `_output/charts` (generated by build):

- To install and set up Helm charts for Rook run `tests/scripts/helm.sh up`.
- To clean up `tests/scripts/helm.sh clean`.

**NOTE:** These helper scripts depend on some artifacts under the `_output/` directory generated during build time.
These scripts should be run from the project root.

**NOTE**: If Helm is not available in your `PATH`, Helm will be downloaded to a temporary directory
(`/tmp/rook-tests-scripts-helm`) and used from that directory.