DROP TABLE IF EXISTS baseball_stats;
CREATE TABLE baseball_stats
SELECT b.game_id, 
	   b.home_runs, 
	   b.home_hits, 
	   b.home_errors,
	   b.away_errors, 
	   p.startingPitcher, 
	   p.endingInning as pitcher_endingInning, 
	   p.pitchesThrown, 
	   p.Strikeout as pitcher_strikeout, 
	   p.Walk as pitcher_walk, 
	   t.toBase as team_toBase, 
	   t.inning as team_inning, 
	   IF(t.atBat>0,(t.Hit*1.0 /t.atBat),0) AS team_batting_avg, 
	   b.winner_home_or_away as HomeTeamWins
FROM  boxscore as b
INNER JOIN team_batting_counts t ON t.game_id = b.game_id
INNER JOIN pitcher_counts p ON p.game_id = t.game_id
GROUP by b.game_id;

ALTER TABLE baseball_stats
  MODIFY team_batting_avg DOUBLE(5,4);
 
ALTER TABLE baseball_stats DROP COLUMN game_id;