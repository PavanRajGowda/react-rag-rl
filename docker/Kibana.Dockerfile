FROM docker.elastic.co/elasticsearch/elasticsearch:8.19.3 AS elasticsearch
FROM docker.elastic.co/kibana/kibana:8.19.3
COPY --from=elasticsearch /usr/share/elasticsearch /usr/share/elasticsearch
USER root
ENTRYPOINT []
