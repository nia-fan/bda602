-- team batting stats
DROP TABLE IF EXISTS game_team_batting_counts_join;
CREATE TABLE game_team_batting_counts_join
SELECT 
		g.game_id,
		DATE(g.local_date) AS local_date,
		g.away_team_id,
		g.home_team_id,
		tbc.team_id,
		tbc.homeTeam,
		tbc.finalScore,
		tbc.opponent_finalScore,
        tbc.atbat,
        tbc.Hit,
        tbc.Home_Run,
        tbc.Ground_Out,
        tbc.plateApperance,
        tbc.Fly_Out,
        tbc.inning,
        tbc.Hit_By_Pitch,
        tbc.Single,
        tbc.Double AS double_play,
        tbc.Triple,
        tbc.Sac_Fly,
        tbc.Strikeout,
        tbc.Intent_Walk,
        tbc.Walk,
        (tbc.Walk - tbc.Intent_Walk) AS NIWalk,
        tbc.Field_Error
    FROM game g
    JOIN team_batting_counts tbc ON g.game_id = tbc.game_id;
   
   CREATE INDEX game_team_batting_counts_join_index ON game_team_batting_counts_join(game_id);
  
--  team_batting_counts rolling
DROP TABLE IF EXISTS game_team_batting_counts_rolling;
CREATE TABLE game_team_batting_counts_rolling
SELECT 
	tbc.game_id,
	tbc.team_id,
	tbc.homeTeam,
	tbc.local_date,
    tbc.finalScore,
    IF(SUM(tbc2.atBat)>0, (SUM(tbc2.Hit) / SUM(tbc2.atBat)),0) AS BA_rolling,
    IF(SUM(tbc2.Home_Run)>0, (SUM(tbc2.atBat) / SUM(tbc2.Home_Run)),0) AS AB_HR_rolling,
	IF(SUM(tbc2.Hit)>0, (SUM(tbc2.Home_Run) / SUM(tbc2.Hit)),0) AS HR_H_rolling,
	IF(SUM(tbc2.Fly_Out)>0, (SUM(tbc2.Ground_Out) / SUM(tbc2.Fly_Out)),0) AS GOAO_rolling,
	IF(SUM(tbc2.atBat)+SUM(tbc2.Walk)+SUM(tbc2.Hit_By_Pitch)+SUM(tbc2.Sac_Fly)>0, (SUM(tbc2.Hit)+SUM(tbc2.Walk)+SUM(tbc2.Hit_By_Pitch))/ (SUM(tbc2.atBat)+SUM(tbc2.Walk)+SUM(tbc2.Hit_By_Pitch)+SUM(tbc2.Sac_Fly)),0) as OBP_rolling,
	IF((SUM(tbc2.atBat)+SUM(tbc2.Walk)-SUM(tbc2.Intent_Walk)+SUM(tbc2.Sac_Fly)+SUM(tbc2.Hit_By_Pitch))>0, (0.69*SUM(tbc2.NIWalk)+0.719*SUM(tbc2.Hit_By_Pitch)+0.87*SUM(tbc2.Single)+1.217*SUM(tbc2.double_play)+1.529*SUM(tbc2.Triple)+1.94*SUM(tbc2.Home_Run))/ (SUM(tbc2.atBat)+SUM(tbc2.Walk)-SUM(tbc2.Intent_Walk)+SUM(tbc2.Sac_Fly)+SUM(tbc2.Hit_By_Pitch)),0) as wOBA_rolling,
	(SUM(tbc2.Single)+2*SUM(tbc2.double_play)+3*SUM(tbc2.Triple)+4*SUM(tbc2.Home_Run)) AS TB_rolling,
	IF(SUM(tbc2.Strikeout)>0, (SUM(tbc2.plateApperance) / SUM(tbc2.Strikeout)),0) AS PA_SO_rolling,
	IF(SUM(tbc2.atBat), (SUM(tbc2.Single)+2*SUM(tbc2.double_play)+3*SUM(tbc2.Triple)+4*SUM(tbc2.Home_Run))/SUM(tbc2.atBat),0) AS SLG_rolling,
	IF(SUM(tbc2.inning)>0, 9*SUM(tbc2.Strikeout) / SUM(tbc2.inning),0) AS K9_rolling,
	IF(SUM(tbc2.inning)>0, (SUM(tbc2.Walk)+SUM(tbc2.Hit)) / SUM(tbc2.inning),0) AS WHIP_rolling,
	IF(SUM(tbc2.Strikeout)>0, SUM(tbc2.Walk) / SUM(tbc2.Strikeout),0) AS BB_K_rolling,
	AVG(tbc2.Home_Run) AS HomeRun_rolling,
	(SUM(tbc2.Hit)+SUM(tbc2.Walk)+SUM(tbc2.Hit_By_Pitch)) AS TOB_rolling,
	AVG(tbc2.Strikeout) AS Strikeout_rolling,
	(SUM(tbc2.finalScore)-SUM(tbc2.opponent_finalScore)) AS RDIFF_rolling,
	AVG(tbc2.Field_Error) AS FieldError_avg_rolling
FROM game_team_batting_counts_join tbc 
JOIN game_team_batting_counts_join AS tbc2
	ON (tbc2.local_date > DATE_ADD(tbc.local_date, INTERVAL - 20 DAY) AND tbc2.local_date < tbc.local_date) AND tbc2.team_id = tbc.team_id
GROUP BY tbc.team_id, tbc.local_date;
	
CREATE INDEX game_team_batting_counts_rollingindex ON game_team_batting_counts_rolling(game_id);

-- seperate home team and away team stats
DROP TABLE IF EXISTS game_team_batting_counts_stats;
CREATE TABLE game_team_batting_counts_stats
SELECT 
	g.game_id,
	g.local_date,
	away_tbc.finalScore AS away_finalScore,
    home_tbc.finalScore AS home_finalScore,
	away_tbc.BA_rolling AS away_BA_rolling,
	home_tbc.BA_rolling AS home_BA_rolling,
	away_tbc.AB_HR_rolling AS away_AB_HR_rolling,
	home_tbc.AB_HR_rolling AS home_AB_HR_rolling,
	away_tbc.HR_H_rolling AS away_HR_H_rolling,
	home_tbc.HR_H_rolling AS home_HR_H_rolling,
	away_tbc.GOAO_rolling AS away_GOAO_rolling,
	home_tbc.GOAO_rolling AS home_GOAO_rolling,
	away_tbc.OBP_rolling AS away_OBP_rolling,
	home_tbc.OBP_rolling AS home_OBP_rolling,
	away_tbc.wOBA_rolling AS away_wOBA_rolling,
	home_tbc.wOBA_rolling AS home_wOBA_rolling,
	away_tbc.TB_rolling AS away_TB_rolling,
	home_tbc.TB_rolling AS home_TB_rolling,
	away_tbc.PA_SO_rolling AS away_PA_SO_rolling,
	home_tbc.PA_SO_rolling AS home_PA_SO_rolling,
	away_tbc.SLG_rolling AS away_SLG_rolling,
	home_tbc.SLG_rolling AS home_SLG_rolling,
	away_tbc.K9_rolling AS away_K9_rolling,
	home_tbc.K9_rolling AS home_K9_rolling,
	away_tbc.WHIP_rolling AS away_WHIP_rolling,
	home_tbc.WHIP_rolling AS home_WHIP_rolling,
	away_tbc.BB_K_rolling AS away_BB_K_rolling,
	home_tbc.BB_K_rolling AS home_BB_K_rolling,
	away_tbc.HomeRun_rolling AS away_HomeRun_rolling,
	home_tbc.HomeRun_rolling AS home_HomeRun_rolling,
	away_tbc.TOB_rolling AS away_TOB_rolling,
	home_tbc.TOB_rolling AS home_TOB_rolling,
	away_tbc.Strikeout_rolling AS away_Strikeout_rolling,
	home_tbc.Strikeout_rolling AS home_Strikeout_rolling,
	away_tbc.RDIFF_rolling AS away_RDIFF_rolling,
	home_tbc.RDIFF_rolling AS home_RDIFF_rolling,
	away_tbc.FieldError_avg_rolling AS away_FieldError_rolling,
	home_tbc.FieldError_avg_rolling AS home_FieldError_rolling
FROM game g
JOIN game_team_batting_counts_rolling away_tbc ON g.away_team_id = away_tbc.team_id AND g.game_id = away_tbc.game_id
JOIN game_team_batting_counts_rolling home_tbc ON g.home_team_id = home_tbc.team_id AND g.game_id = home_tbc.game_id;


CREATE INDEX game_team_batting_counts_stats_index ON game_team_batting_counts_stats(game_id);

-- get info from pitcher_counts
DROP TABLE IF exists game_pc_join;
CREATE TABLE game_pc_join
SELECT
	g.game_id,
	DATE(g.local_date) AS local_date,
	g.away_team_id,
	g.home_team_id,
	pc.team_id,
	pc.startingPitcher,
	pc.pitchesThrown
FROM game g 
JOIN  pitcher_counts pc on g.game_id = pc.game_id 
GROUP BY game_id, team_id;

CREATE INDEX game_pc_join_index ON game_pc_join(game_id);

DROP TABLE IF EXISTS gpc_rolling;
CREATE TABLE gpc_rolling
SELECT 
	gpc.game_id,
	gpc.local_date,
	gpc.team_id,
	AVG(gpc2.startingPitcher) AS startingPitcher_rolling,
	AVG(gpc2.pitchesThrown) AS pitchesThrown_rolling
FROM game_pc_join gpc
JOIN game_pc_join AS gpc2
	ON (gpc2.local_date > DATE_ADD(gpc.local_date, INTERVAL - 20 DAY) AND gpc2.local_date < gpc.local_date) and gpc2.team_id = gpc.team_id
GROUP BY gpc.team_id, gpc.local_date;

CREATE INDEX gpc_rolling_index ON gpc_rolling(game_id);

DROP TABLE IF EXISTS gpc_stats;
CREATE TABLE gpc_stats
SELECT 
	g.game_id,
	away_gpc.startingPitcher_rolling as away_startingPitcher_rolling,
	home_gpc.startingPitcher_rolling as home_startingPitcher_rolling,
	away_gpc.pitchesThrown_rolling as away_pitchesThrown_rolling,
	home_gpc.pitchesThrown_rolling as home_pitchesThrown_rolling
FROM game g
JOIN gpc_rolling away_gpc ON g.away_team_id = away_gpc.team_id AND g.game_id = away_gpc.game_id
JOIN gpc_rolling home_gpc ON g.home_team_id = home_gpc.team_id AND g.game_id = home_gpc.game_id;


CREATE INDEX gpc_stats_index ON gpc_stats(game_id);

DROP TABLE IF EXISTS game_stats_temp;
CREATE TABLE game_stats_temp
SELECT gtbcs.*,
	gpcs.away_startingPitcher_rolling,
	gpcs.home_startingPitcher_rolling,
	gpcs.away_pitchesThrown_rolling,
	gpcs.home_pitchesThrown_rolling
FROM game_team_batting_counts_stats gtbcs
JOIN gpc_stats gpcs on gtbcs.game_id = gpcs.game_id;


DROP TABLE IF EXISTS baseball_game_stats;
CREATE TABLE baseball_game_stats
SELECT gst.*,
	(gst.home_K9_rolling - gst.home_Strikeout_rolling) as home_K9_StrikeOut_Diff,
	(gst.away_K9_rolling - gst.away_Strikeout_rolling) as away_K9_StrikeOut_Diff,
	(gst.home_OBP_rolling - gst.home_WHIP_rolling) as home_OBP_WHIP_Diff,
	(gst.away_OBP_rolling - gst.away_WHIP_rolling) as away_OBP_WHIP_Diff,
	(gst.away_TB_rolling - gst.away_TOB_rolling) as away_TB_TOB_Diff,
	(gst.home_TB_rolling - gst.home_TOB_rolling) as home_TB_TOB_Diff,
	(gst.home_wOBA_rolling - gst.home_SLG_rolling) as home_wOBA_SLG_Diff,
	(gst.away_wOBA_rolling - gst.away_SLG_rolling) as away_wOBA_SLG_Diff,
	
	(gst.away_HR_H_rolling - gst.away_HomeRun_rolling) as away_HR_HomeRun_Diff,
	(gst.home_HR_H_rolling - gst.home_HomeRun_rolling) as home_HR_HomeRun_Diff,
	(gst.home_OBP_rolling - gst.home_wOBA_rolling) as home_OBP_wOBA_Diff,
	(gst.away_OBP_rolling - gst.away_wOBA_rolling) as away_OBP_wOBA_Diff,
	SUBSTRING_INDEX(b.wind,' ',1) AS wind_MPH,
	SUBSTRING_INDEX(b.temp,' ',1) AS temp_degree,
	(CASE WHEN gst.away_finalScore>gst.home_finalScore THEN "A"
      WHEN gst.away_finalScore<gst.home_finalScore THEN "H"
      WHEN gst.away_finalScore=gst.home_finalScore THEN "T"
      END) AS HomeTeamWins
FROM game_stats_temp gst
JOIN boxscore b on gst.game_id = b.game_id and b.wind <> "Indoors";

ALTER TABLE baseball_game_stats
  MODIFY away_BA_Rolling DOUBLE(8,6),
  MODIFY home_BA_Rolling DOUBLE(8,6),
  MODIFY away_AB_HR_rolling DOUBLE(8,4),
  MODIFY home_AB_HR_rolling DOUBLE(8,4),
  MODIFY away_HR_H_rolling DOUBLE(8,6),
  MODIFY home_HR_H_rolling DOUBLE(8,6),
  MODIFY away_GOAO_rolling DOUBLE(8,6),
  MODIFY home_GOAO_rolling DOUBLE(8,6),
  MODIFY away_OBP_rolling DOUBLE(8,6),
  MODIFY home_OBP_rolling DOUBLE(8,6),
  MODIFY away_wOBA_rolling DOUBLE(8,6),
  MODIFY home_wOBA_rolling DOUBLE(8,6),
  MODIFY away_TB_rolling DOUBLE(8,2),
  MODIFY home_TB_rolling DOUBLE(8,2),
  MODIFY away_PA_SO_rolling DOUBLE(8,6),
  MODIFY home_PA_SO_rolling DOUBLE(8,6),
  MODIFY away_SLG_rolling DOUBLE(8,6),
  MODIFY home_SLG_rolling DOUBLE(8,6), 
  MODIFY away_K9_rolling DOUBLE(8,6),
  MODIFY home_K9_rolling DOUBLE(8,6), 
  MODIFY away_WHIP_rolling DOUBLE(8,6),
  MODIFY home_WHIP_rolling DOUBLE(8,6),
  MODIFY away_BB_K_rolling DOUBLE(8,6),
  MODIFY home_BB_K_rolling DOUBLE(8,6),
  MODIFY away_HomeRun_rolling DOUBLE(8,6),
  MODIFY home_HomeRun_rolling DOUBLE(8,6),
  MODIFY away_TOB_rolling DOUBLE(8,2),
  MODIFY home_TOB_rolling DOUBLE(8,2),
  MODIFY away_Strikeout_rolling DOUBLE(8,6),
  MODIFY home_Strikeout_rolling DOUBLE(8,6),
  MODIFY away_RDIFF_rolling DOUBLE(8,2),
  MODIFY home_RDIFF_rolling DOUBLE(8,2),
  MODIFY away_FieldError_rolling DOUBLE(8,6),
  MODIFY home_FieldError_rolling DOUBLE(8,6),
  MODIFY away_startingPitcher_rolling DOUBLE(8,6),
  MODIFY home_startingPitcher_rolling DOUBLE(8,6),
  MODIFY away_pitchesThrown_rolling DOUBLE(8,5),
  MODIFY home_pitchesThrown_rolling DOUBLE(8,5),
  MODIFY home_K9_StrikeOut_Diff DOUBLE(8,5),
  MODIFY away_K9_StrikeOut_Diff DOUBLE(8,5),
  MODIFY home_OBP_WHIP_Diff DOUBLE(8,5),
  MODIFY away_OBP_WHIP_Diff DOUBLE(8,5),
  MODIFY away_TB_TOB_Diff DOUBLE(8,5),
  MODIFY home_TB_TOB_Diff DOUBLE(8,5),
  MODIFY home_wOBA_SLG_Diff DOUBLE(8,5),
  MODIFY away_wOBA_SLG_Diff DOUBLE(8,5),
  MODIFY away_HR_HomeRun_Diff DOUBLE(8,5),
  MODIFY home_HR_HomeRun_Diff DOUBLE(8,5),
  MODIFY home_OBP_wOBA_Diff DOUBLE(8,5),
  MODIFY away_OBP_wOBA_Diff DOUBLE(8,5),
  MODIFY wind_MPH DOUBLE(6,2),
  MODIFY temp_degree DOUBLE(6,2);

 
 

 
 