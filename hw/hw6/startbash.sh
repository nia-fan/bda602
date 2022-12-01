#!/bin/sh

DATABASE_USER=root
DATABASE_PWD=xxxxx
DATABASE_NAME=baseball

# if mysql -h 172.28.0.2 -P 3306 -protocol=TCP -u$DATABASE_USER -p$DATABASE_PWD -e "use $DATABASE_NAME";
if mysql -h172.17.0.2 -P3306 -protocol=TCP -u$DATABASE_USER -p$DATABASE_PWD -e "use $DATABASE_NAME";
then
echo "Database $DATABASE_NAME already exists. Exiting."
else
echo Create database
mysql -h172.17.0.2 -P3306 -protocol=TCP -u$DATABASE_USER -p$DATABASE_PWD -e "CREATE DATABASE $DATABASE_NAME"
echo Load Data
mysql -h172.17.0.2 -P3306 -protocol=TCP -u$DATABASE_USER -p$DATABASE_PWD $DATABASE_NAME < /baseball.sql
fi
echo run SQL file
sleep 10
mysql -h172.17.0.2 -P3306 -protocol=TCP -u$DATABASE_USER -p$DATABASE_PWD $DATABASE_NAME < /hw6.sql
