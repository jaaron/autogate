FROM pymesh/pymesh
RUN pip install bezier matplotlib
COPY autogate-pymesh.py /usr/local/bin
RUN ["chmod", "+x", "/usr/local/bin/autogate-pymesh.py"]
ENTRYPOINT ["/usr/local/bin/autogate-pymesh.py"]
