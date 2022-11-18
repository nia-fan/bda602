
DROP TABLE IF EXISTS baseball_stats;

CREATE TABLE baseball_stats
SELECT b.game_id, b.home_runs, b.home_hits, home_errors, p.startingPitcher, p.toBase as pitcherToBase, p.pitchesThrown, p.Strikeout, 
	   p.Walk as pitcherWalk, t.atBat as teamatBat, t.inning as teammInning, b.winner_home_or_away as HomeTeamWins
FROM  boxscore as b
INNER JOIN team_batting_counts t ON t.game_id = b.game_id
INNER JOIN pitcher_counts p ON p.game_id = t.game_id
GROUP by b.game_id;

ALTER TABLE baseball_stats DROP COLUMN game_id;
