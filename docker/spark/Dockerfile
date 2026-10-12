# bitnami/spark was removed from Docker Hub; legacy mirror keeps 3.5.3 tags.
FROM bitnamilegacy/spark:3.5.3

USER root

# Legacy image runs as uid 1001 without a passwd entry; Hadoop/PySpark need one.
RUN if ! getent passwd 1001 >/dev/null; then \
		echo 'spark:x:1001:0:Spark:/opt/bitnami/spark:/bin/bash' >> /etc/passwd; \
	fi

# Use Bitnami's Spark venv (ships pyspark + py4j). Install only app deps there.
COPY requirements.txt /tmp/requirements.txt
RUN /opt/bitnami/spark/venv/bin/pip install --no-cache-dir \
		kafka-python \
		requests \
		numpy \
		pandas \
	&& mkdir -p /opt/bitnami/spark/.ivy2 /opt/project \
	&& chown -R 1001:0 /opt/bitnami/spark/.ivy2 /opt/project /opt/bitnami/spark/venv

ENV PATH=/opt/bitnami/spark/venv/bin:/opt/bitnami/spark/bin:$PATH \
	PYTHONPATH=/opt/bitnami/spark/python:/opt/project/src \
	SPARK_HOME=/opt/bitnami/spark \
	HOME=/opt/bitnami/spark \
	IVY_HOME=/opt/bitnami/spark/.ivy2 \
	SPARK_VERSION=3.5.3

WORKDIR /opt/project

USER 1001
