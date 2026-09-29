#!/usr/bin/env bash
# Build every video clip and poster for the PACT project site (see MEDIA MANIFEST).
# Usage: tools/make_media.sh [section ...]   (run from anywhere; outputs go to static/media/ and static/viewer/)
#   sections: main hero annotation world beyond forcewall viewer   (default: all)
set -euo pipefail

FFMPEG=${FFMPEG:-/opt/homebrew/bin/ffmpeg}
SITE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$SITE/static/media"
VIEW="$SITE/static/viewer"
SRC_ROOT=/Users/rikhat.akizhanov/Desktop/projects/Climbing
SRC="$SRC_ROOT/VideoReconstruction"
RES="$SRC/results"
R3U="$SRC/results_3d_update"   # 2304x768 triptychs: input + PACT overlay | 3D camera view | 3D side view (no labels)
WILD="$SRC/wild"
ANN="$SRC_ROOT/supmat_video/assets/annotation"
SUPMAT="$SRC_ROOT/supmat_video/PACT_supmat.mp4"

SECTIONS=" ${*:-main hero annotation world beyond forcewall viewer} "
want() { [[ $SECTIONS == *" $1 "* ]]; }

mkdir -p "$OUT"/{main,hero,annotation,world,beyond,forcewall}

ENC=(-an -c:v libx264 -pix_fmt yuv420p -crf 24 -preset slow -movflags +faststart -r 30)
EVEN="scale=trunc(iw/2)*2:trunc(ih/2)*2"
PORTRAIT="scale=540:960:force_original_aspect_ratio=increase,crop=540:960"

poster() {  # $1 = clip.mp4 -> clip.jpg (first frame, same size)
  "$FFMPEG" -hide_banner -loglevel error -y -i "$1" -frames:v 1 -q:v 3 "${1%.mp4}.jpg"
}

# clip <src> <dst> <start|""> <duration|""> <filter>
clip() {
  local src=$1 dst=$2 ss=$3 dur=$4 vf=$5
  local trim=()
  [[ -n $ss ]] && trim+=(-ss "$ss")
  [[ -n $dur ]] && trim+=(-t "$dur")
  echo ">> $dst"
  "$FFMPEG" -hide_banner -loglevel error -y ${trim[@]+"${trim[@]}"} -i "$src" -vf "$vf" "${ENC[@]}" "$dst"
  poster "$dst"
}

# pane <src> <dst> <rate> : input footage for a 3D viewer. Keeps EVERY decoded frame (the 3D rows are
# indexed by source frame): no -r, no trim; timestamps are rewritten to a constant <rate> (the source's
# nominal rate) so frame i starts exactly at i/<rate>. Longer side scaled to 720.
pane() {
  local src=$1 dst=$2 rate=$3
  echo ">> $dst"
  mkdir -p "$(dirname "$dst")"
  "$FFMPEG" -hide_banner -loglevel error -y -i "$src" \
    -vf "setpts=N/($rate)/TB,scale='if(gt(iw,ih),720,-2)':'if(gt(iw,ih),-2,720)'" \
    -fps_mode passthrough -enc_time_base "1/($rate)" \
    -an -c:v libx264 -pix_fmt yuv420p -crf 24 -preset slow -movflags +faststart "$dst"
  poster "$dst"
}

# ---- main supplementary video (keeps audio) ----
if want main; then
echo ">> $OUT/main/pact.mp4"
"$FFMPEG" -hide_banner -loglevel error -y -i "$SUPMAT" \
  -vf "scale=1920:1080,$EVEN" -c:v libx264 -pix_fmt yuv420p -crf 22 -preset slow \
  -c:a aac -b:a 128k -movflags +faststart "$OUT/main/pact.mp4"
"$FFMPEG" -hide_banner -loglevel error -y -ss 3 -i "$OUT/main/pact.mp4" -frames:v 1 -q:v 3 "$OUT/main/pact.jpg"
fi

# ---- hero strip (portrait 540x960) ----
if want hero; then
clip "$RES/climbing_dyno.mp4"      "$OUT/hero/strip_1.mp4" 11.0 7.0 "$PORTRAIT"
clip "$RES/yoga_1.mp4"             "$OUT/hero/strip_2.mp4" 1.0  8.0 "$PORTRAIT"
clip "$RES/20240327_163731_1.mp4"  "$OUT/hero/strip_3.mp4" 15.5 7.0 "crop=600:1068:423:238,$PORTRAIT"
# strip_4: the athlete walks left and back (crop centre ~1150 -> 650 -> 1030 px), so a fixed 608 px crop
# cuts him off; a slow cosine pan (period 10 s) keeps him inside. y starts at 110 to drop the burned-in
# "SQUAT WALKS" title (rows 50-100, x<=515), so the 9:16 window is 546x970.
clip "$RES/full_body_2.mp4"        "$OUT/hero/strip_4.mp4" 0.0  9.5 "crop=w=546:h=970:x='642+200*cos(2*PI*t/10)':y=110,$PORTRAIT"
clip "$RES/competition_1.mp4"      "$OUT/hero/strip_5.mp4" 2.0  8.0 "$PORTRAIT"
fi

# ---- annotation pipeline on boulder_1: 5 stage clips (720x1280 -> 540x960, full 14.83 s, in sync) ----
if want annotation; then
STAGE="scale=540:960,$EVEN"
clip "$ANN/boulder_1_sam3.mp4"           "$OUT/annotation/stage_sam3.mp4"    "" "" "$STAGE"
clip "$ANN/boulder_1_sapiens.mp4"        "$OUT/annotation/stage_sapiens.mp4" "" "" "$STAGE"
clip "$ANN/boulder_1_body.mp4"           "$OUT/annotation/stage_body.mp4"    "" "" "$STAGE"
clip "$ANN/boulder_1_scene.mp4"          "$OUT/annotation/stage_scene.mp4"   "" "" "$STAGE"
clip "$ANN/boulder_1_labels_overlay.mp4" "$OUT/annotation/stage_labels.mp4"  "" "" "$STAGE"
# physics-based label triptychs (2304x768 -> 1728x576, full length)
TRI="scale=1728:576,$EVEN"
clip "$ANN/boulder_1_labels.mp4" "$OUT/annotation/labels_boulder_1.mp4" "" "" "$TRI"
clip "$ANN/boulder_3_labels.mp4" "$OUT/annotation/labels_boulder_3.mp4" "" "" "$TRI"
fi

# ---- model results: results_3d_update triptychs (2304x768 -> 1728x576) ----
# Input panels were QA'd on contact sheets for burned-in text/logos; see the notes per clip.
TRI="scale=1728:576,$EVEN"
if want world; then
clip "$R3U/20230917_114335.mp4"   "$OUT/world/climb_2023.mp4"    16.0 14.0 "$TRI"
clip "$R3U/20240327_163731_1.mp4" "$OUT/world/climb_2024.mp4"    8.0  11.0 "$TRI"
# 2.0-16.0 s: the window the supmat comparison used (floor start, then the steep overhang section)
clip "$R3U/20250217_203915.mp4"   "$OUT/world/climb_2025.mp4"    2.0  14.0 "$TRI"
clip "$R3U/competition_1.mp4"     "$OUT/world/competition_1.mp4" 2.0  10.9 "$TRI"
clip "$R3U/climbing_dyno.mp4"     "$OUT/world/climbing_dyno.mp4" 9.0  9.0  "$TRI"
fi

if want beyond; then
clip "$R3U/olympics_2.mp4"      "$OUT/beyond/olympics_2.mp4"      8.0 11.0 "$TRI"
clip "$R3U/yoga_1.mp4"          "$OUT/beyond/yoga_1.mp4"          ""  ""   "$TRI"
clip "$R3U/box_step.mp4"        "$OUT/beyond/box_step.mp4"        ""  ""   "$TRI"
clip "$R3U/backflip.mp4"        "$OUT/beyond/backflip.mp4"        ""  ""   "$TRI"
# the input panel of full_body_2 is a crop that already excludes the source's burned-in title/counter
clip "$R3U/full_body_2.mp4"     "$OUT/beyond/full_body_2.mp4"     0.0 9.5  "$TRI"
clip "$R3U/aerial.mp4"          "$OUT/beyond/aerial.mp4"          ""  ""   "$TRI"
clip "$R3U/back_handspring.mp4" "$OUT/beyond/back_handspring.mp4" ""  ""   "$TRI"
clip "$R3U/back_walkover.mp4"   "$OUT/beyond/back_walkover.mp4"   ""  ""   "$TRI"
# broad_jump is NOT used: its input panel carries an "@BROTHERFARIS" watermark (and later an Instagram
# logo) from ~0.9 s of its 2 s. Swapped for tic_tac (clean except a few-pixel sliver of a coloured bar
# at the bottom-right edge for ~0.5 s). frontflip ("Bob Reese" watermark) and full_body_3 ("JACKS"
# title) were rejected as swaps.
clip "$R3U/tic_tac.mp4"         "$OUT/beyond/tic_tac.mp4"         ""  ""   "$TRI"
fi

# ---- ForceWall validation video (1898x1252 -> width 1440, full length) ----
if want forcewall; then
clip "$SRC_ROOT/results/climb_wall_3_a_16_03_part1.mp4" "$OUT/forcewall/climb_wall.mp4" "" "" "scale=1440:-2"
fi

# ---- 3D viewer video panes (every frame, source rate) ----
if want viewer; then
pane "$SRC/boulder_1/boulder_1.mp4"  "$VIEW/annotation/boulder_1/video.mp4"     30
pane "$SRC/boulder_3/boulder_3.mp4"  "$VIEW/annotation/boulder_3/video.mp4"     2997/100
pane "$WILD/climbing_dyno.mp4"       "$VIEW/prediction/climbing_dyno/video.mp4" 30
pane "$WILD/backflip.mp4"            "$VIEW/prediction/backflip/video.mp4"      30000/1001
# the raw olympics_2 input is not in wild/; videos/olympics_2.mp4 decodes to the same 476 frames as the
# PACT renders. Its nominal rate says 25 but the frames span 4997/200 fps (the rate PACT ran at).
pane "$SRC_ROOT/videos/olympics_2.mp4" "$VIEW/prediction/olympics_2/video.mp4"  4997/200
pane "$WILD/yoga_1.mp4"              "$VIEW/prediction/yoga_1/video.mp4"        30
fi

echo "done."
