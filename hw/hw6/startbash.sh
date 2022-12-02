#!/bin/sh

DATABASE_USER=root
DATABASE_PWD=xxxx
DATABASE_NAME=baseball
sleep 5

if ! mariadb -u $DATABASE_USER -p$DATABASE_PWD -h mariadb -e "USE $DATABASE_NAME;"
then
    echo Create database
    mariadb -u $DATABASE_USER -p$DATABASE_PWD -h mariadb -e "CREATE DATABASE $DATABASE_NAME;"
    echo Load Data
    mariadb -u $DATABASE_USER -p$DATABASE_PWD -h mariadb -D $DATABASE_NAME < baseball.sql
else
    echo "Database $DATABASE_NAME already exists."
fi
sleep 2
echo run SQL file
mariadb -u $DATABASE_USER -p$DATABASE_PWD -h mariadb -D $DATABASE_NAME < hw6.sql

sleep 2
echo write out put files
mariadb -u $DATABASE_USER -p$DATABASE_PWD -h mariadb $DATABASE_NAME -e "select * FROM rolling_avg_100_days;" > /results/ba_rolling_avg.txt
mariadb -u $DATABASE_USER -p$DATABASE_PWD -h mariadb $DATABASE_NAME -e "select * FROM ba_game_12560;" > /results/ba_game_12560.txt
mariadb -u $DATABASE_USER -p$DATABASE_PWD -h mariadb $DATABASE_NAME -e "select * FROM ba_game_12560_rolling;" > /results/ba_game_12560_rolling.txt
echo Done!


