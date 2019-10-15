FROM gregunz/jupyterlab:latest

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install --upgrade -r /tmp/requirements.txt


WORKDIR /opt/app/projects
EXPOSE 8888
CMD ["/usr/bin/zsh", "/tmp/run_jupyter.sh"]