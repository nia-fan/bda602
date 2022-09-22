-- calculate each player's historic batting average 
-- calculate avg for each player for each game, then group by batter and avg all 
-- calculate batting average: need to check if atBat is 0, the batterin_average should be 0
DROP TABLE IF EXISTS batting_average_historic;
CREATE TABLE batting_average_historic(
    batter INT(11),
    batting_average_historic DECIMAL(8,6)
);

insert into batting_average_historic(batter,batting_average_historic)
SELECT bc.batter, round(avg(IF(bc.atBat>0,round(bc.Hit*1.0 /bc.atBat,2),0)),4) AS batting_average_historic
FROM  batter_counts as bc 
GROUP BY bc.batter;


-- each player annual batting average
-- join table batter_counts with table game and group by batter and year
-- calculate batting average: need to check if the atBat is 0, the batterin_average should be 0
-- Calculate the batting average for every player: Annual
DROP TABLE IF EXISTS batting_average_annual;
CREATE TABLE batting_average_annual(
    batter INT(11),
    local_date_year year,
    batting_average_annual DECIMAL(8,6)
);
insert into batting_average_annual(batter,local_date_year, batting_average_annual)
SELECT bc.batter, YEAR(g.local_date) as local_date_year, round(avg(IF(bc.atBat>0,round(bc.Hit*1.0 /bc.atBat,2),0)),4) AS batting_average_annual
FROM  batter_counts as bc 
JOIN game as g on bc.game_id = g.game_id 
GROUP BY bc.batter,YEAR(g.local_date)
order by bc.batter;


-- batting Average for the last 100 days that player was in prior to this game
-- drop  batting_game_join table if it exists
DROP TABLE IF EXISTS batting_game_join;
-- Create temp table batting_game_join
CREATE temporary TABLE batting_game_join(
    batter INT(11),
    game_id INT(11),
    atBat INT(11),
    hit INT(11),
    local_date DATE,
    batting_average DECIMAL(8,6)
);

-- Join batter_counts and game based on game_id and group by batter and local_date
-- calculate batting average for each player and each game
-- insert into table batting_game_join
insert into batting_game_join(batter,game_id, atBat, hit, local_date, batting_average)
select bc.batter,bc.game_id,bc.atBat, bc.hit, DATE(g.local_date), round(avg(IF(bc.atBat>0,round(bc.Hit*1.0 /bc.atBat,4),0)),4) as batting_average
from batter_counts as bc 
join game as g 
on g.game_id = bc.game_id
group by batter, DATE(g.local_date)
order by bc.batter, DATE(g.local_date) desc;


-- query latest game date first for each player from batting_game_join
-- then join batting_game_join to query last 100 days that player was in prior to this game
-- calculate batting average 
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

