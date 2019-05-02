FROM elasticsearch:5.3.1

COPY es-plugin/target/releases/esplugin-0.0.1.zip /
RUN /usr/share/elasticsearch/bin/elasticsearch-plugin install file:///esplugin-0.0.1.zip

RUN chown -R elasticsearch:elasticsearch /usr/share/elasticsearch/data
RUN chown -R elasticsearch:elasticsearch /usr/share/elasticsearch/logs

EXPOSE 9200 9300