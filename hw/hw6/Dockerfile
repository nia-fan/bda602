FROM mariadb:latest

ENV APP_HOME /hw6	
WORKDIR $APP_HOME

COPY baseball.sql /docker-entrypoint-initdb.d/
COPY baseball.sql /
COPY hw6.sql /
COPY startbash.sh /

ENV MYSQL_ROOT_PASSWORD mypass22
ENV MYSQL_DATABASE mariadb
ENV MYSQL_USER root
ENV MYSQL_PASSWORD xxxx

# EXPOSE 3306
EXPOSE 3306

CMD ["/startbash.sh"]