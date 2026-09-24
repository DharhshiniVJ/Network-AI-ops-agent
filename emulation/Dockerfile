FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Mininet, OpenvSwitch, and networking utilities
RUN apt-get update && apt-get install -y \
    mininet \
    openvswitch-switch \
    openvswitch-testcontroller \
    iproute2 \
    iputils-ping \
    iperf \
    tcpdump \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Fix Mininet controller issue in newer Ubuntu
RUN ln -s /usr/bin/ovs-testcontroller /usr/bin/controller

WORKDIR /emulation
COPY mininet_topo.py /emulation/

# Start OVS service and run the python script
CMD service openvswitch-switch start && python3 mininet_topo.py
