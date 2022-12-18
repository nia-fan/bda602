#!/bin/sh

DATABASE_USER=root
DATABASE_PWD=xxxxx
DATABASE_NAME=baseball

if ! mariadb -u $DATABASE_USER -p$DATABASE_PWD -h mariadb -e "USE $DATABASE_NAME;"
then
    echo Create database
    mariadb -u $DATABASE_USER -p$DATABASE_PWD -h mariadb -e "CREATE DATABASE $DATABASE_NAME;"
    echo Load Data
    mariadb -u $DATABASE_USER -p$DATABASE_PWD -h mariadb -D $DATABASE_NAME < baseball.sql
else
    echo "Database $DATABASE_NAME already exists."
fi

sleep 1
echo run SQL file
mariadb -u $DATABASE_USER -p$DATABASE_PWD -h mariadb -D $DATABASE_NAME < /sql/baseballstats.sql
echo Run SQL Done!

sleep 1
echo run python
exec python3 /app/final.py

echo Done!


