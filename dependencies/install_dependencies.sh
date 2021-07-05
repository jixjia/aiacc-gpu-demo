#!/bin/bash

set -x
set -e
export OS_TYPE=$(cat /etc/os-release |grep ^ID=|cut -d '=' -f 2|cut -d '"' -f 2)
export OS_VER=$(cat /etc/os-release |grep VERSION_ID=|cut -d '=' -f 2|cut -d '"' -f 2)
if [ "$OS_TYPE" = "ubuntu" ]; then
    dpkg -i openmpi_4.0.1-1_amd64.deb
    mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real
    echo '#!/bin/bash' > /usr/local/bin/mpirun
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun
    chmod a+x /usr/local/bin/mpirun

    mkdir -p /root/.openmpi
    cp -rp default-mca-params.conf /root/.openmpi/mca-params.conf

    apt-get update
    apt-get install -y curl openssh-client openssh-server wget

elif [ "$OS_TYPE" = "centos" ]; then
    yum -y install epel-release
    yum -y install perl openssh-clients openssh-server openblas-devel wget

    rpm -Uivh openmpi-4.0.1-1.el7.x86_64.rpm
    mv /usr/bin/mpirun /usr/bin/mpirun.real
    echo '#!/bin/bash' > /usr/bin/mpirun
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/bin/mpirun
    chmod a+x /usr/bin/mpirun

    PATH=/usr/local/bin:${PATH}
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

    mkdir -p /root/.openmpi
    cp -rp default-mca-params.conf /root/.openmpi/mca-params.conf
fi
