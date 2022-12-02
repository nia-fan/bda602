-- BA for all the players in game_id = 12560
DROP TABLE IF EXISTS ba_game_12560;
CREATE TABLE ba_game_12560
SELECT bc.batter, 
       bc.team_id, 
       bc.game_id,
       bc.Hit,
       bc.atBat, 
       IF(bc.atBat>0, (bc.Hit*1.0 /bc.atBat),0) as Batting_Average
FROM   batter_counts bc 
where bc.game_id = 12560;

-- -- Join the table batter_counts and game based on game_id and group by batter and local_date
DROP TABLE IF EXISTS batting_game_join;
CREATE TABLE batting_game_join
select bc.batter,
	bc.game_id,
	bc.team_id,
	bc.atBat, 
	bc.hit, 
	DATE(g.local_date) as local_date
from batter_counts as bc 
join game as g 
on g.game_id = bc.game_id
group by batter, DATE(g.local_date)
order by bc.batter, DATE(g.local_date) desc;

CREATE INDEX batter_game_index ON batting_game_join(batter, game_id);

-- rolling BA for 100days
DROP TABLE IF EXISTS rolling_avg_100_days;
CREATE TABLE rolling_avg_100_days
SELECT
   bg1.game_id, 
   bg1.team_id,
   bg1.batter,
   bg1.local_date,
   ( SELECT IF(SUM(bg2.atBat)>0, (SUM(bg2.Hit) / SUM(bg2.atBat)),0)
       FROM batting_game_join AS bg2
       where bg1.batter = bg2.batter and (bg2.local_date > DATE_ADD(bg1.local_date, INTERVAL - 100 DAY) and bg2.local_date < bg1.local_date)
       ) AS last_100_days_ba
FROM batting_game_join AS bg1
ORDER BY bg1.local_date desc;

-- rolling BA for all the players in game_id = 12560
DROP TABLE IF EXISTS ba_game_12560_rolling;
CREATE TABLE ba_game_12560_rolling
SELECT batter, 
       team_id, 
       game_id,
       last_100_days_ba
FROM   rolling_avg_100_days
WHERE game_id = 12560;


