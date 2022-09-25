-- Create table batting_game_join
-- drop  batting_game_join table if it exists
DROP TABLE IF EXISTS batting_game_join;
CREATE temporary TABLE batting_game_join(
    batter INT(11),
    game_id INT(11),
    atBat INT(11),
    hit INT(11),
    local_date DATE,
    batting_average DECIMAL(8,6)
);

-- Join the tables batter_counts and game based on game_id, group by batter and local_date (remove time)
-- calculate the batting _average for each player and each game:need to check if the atBat is 0, the batterin_average should be 0
-- insert into table batting_game_join
insert into batting_game_join(batter,game_id, atBat, hit, local_date, batting_average)
select bc.batter,bc.game_id,bc.atBat, bc.hit, DATE(g.local_date), round(avg(IF(bc.atBat>0,round(bc.Hit*1.0 /bc.atBat,4),0)),4) as batting_average
from batter_counts as bc 
join game as g 
on g.game_id = bc.game_id
group by batter, DATE(g.local_date)
order by bc.batter, DATE(g.local_date) desc;


-- calculate each player's historic batting_avg 
-- group by batter and calculate historic batting_average 
-- from batting_game_join
DROP TABLE IF EXISTS batting_average_historic;
CREATE TABLE batting_average_historic(
    batter INT(11),
    batting_average_historic DECIMAL(8,6)
);
insert into batting_average_historic(batter,batting_average_historic)
SELECT bg.batter, avg(Batting_Average) AS batting_average_historic
FROM  batting_game_join as bg
GROUP BY bg.batter;


-- each player's annual batting_avg
-- group by batter and year and calculate annual batting_average
-- from batting_game_join
DROP TABLE IF EXISTS batting_average_annual;
CREATE TABLE batting_average_annual(
    batter INT(11),
    local_date_year year,
    batting_average_annual DECIMAL(8,6)
);
insert into batting_average_annual(batter,local_date_year,batting_average_annual)
select bg.batter, YEAR(bg.local_date) as local_date_year, round(avg(bg.batting_average),4) AS batting_average_annual
from  batting_game_join as bg 
group by bg.batter,YEAR(bg.local_date)
order by bg.batter;


-- pivot 
-- create a pivot table for each player's annual batting_avg
DROP TABLE IF EXISTS batting_average_annual_pivot;
CREATE TABLE batting_average_annual_pivot(
    batter INT(11),
    2007_batting_avg DECIMAL(8,6),
    2008_batting_avg DECIMAL(8,6),
    2009_batting_avg DECIMAL(8,6),
    2010_batting_avg DECIMAL(8,6),
    2011_batting_avg DECIMAL(8,6),
    2012_batting_avg DECIMAL(8,6)
);

insert into batting_average_annual_pivot(batter,2007_batting_avg, 2008_batting_avg,2009_batting_avg,2010_batting_avg,2011_batting_avg,2012_batting_avg)
select
    batter,
    coalesce(sum(case when local_date_year = "2007" then batting_average_annual end), 0) as 2007_batting_avg,
    coalesce(sum(case when local_date_year = "2008" then batting_average_annual end), 0) as 2008_batting_avg,
    coalesce(sum(case when local_date_year = "2009" then batting_average_annual end), 0) as 2009_batting_avg,
    coalesce(sum(case when local_date_year = "2010" then batting_average_annual end), 0) as 2010_batting_avg,
    coalesce(sum(case when local_date_year = "2011" then batting_average_annual end), 0) as 2011_batting_avg,
    coalesce(sum(case when local_date_year = "2012" then batting_average_annual end), 0) as 2012_batting_avg
from batting_average_annual
group by batter;



-- batting _average for the last 100 days that player was in prior to this game
-- query latest game date first for each player from batting_game_join
-- then join batting_game_join to query for the last 100 days that player was in prior to this game
-- calculate batting_average 
DROP TABLE IF EXISTS batting_average_last_100days;
CREATE TABLE batting_average_last_100days(
    batter INT(11),
    batting_avg_last_100Days DECIMAL(8,6)
);
insert into batting_average_last_100days(batter,batting_avg_last_100Days)
select bg.batter, round(avg(bg.batting_average),4) as batting_avg_last_100Days
from batting_game_join as bg
inner join 
    (select batter, max(local_date) as local_date_max
         from batting_game_join 
         group by batter) last_game
on last_game.batter = bg.batter and (bg.local_date between last_game.local_date_max - INTERVAL 100 DAY and last_game.local_date_max)
group by bg.batter;