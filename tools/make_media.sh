#!/usr/bin/env bash
# Build every video clip and poster for the PACT project site (see MEDIA MANIFEST).
# Usage: tools/make_media.sh   (run from anywhere; outputs go to static/media/ of this repo)
set -euo pipefail

FFMPEG=${FFMPEG:-/opt/homebrew/bin/ffmpeg}
SITE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$SITE/static/media"
SRC=/Users/rikhat.akizhanov/Desktop/projects/Climbing/VideoReconstruction
RES="$SRC/results"
R3D="$SRC/results_3d"
WILD="$SRC/wild"
SUPMAT="/Users/rikhat.akizhanov/Desktop/projects/Climbing/supmat_video/ PACT_supmat.mp4"  # leading space is intentional

mkdir -p "$OUT"/{main,hero,world,beyond,overlay,toggle}

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

# pair <src> <dst> <start|""> <duration|"">  : results_3d panels 0 (video) and 2 (side view), labels cropped off
pair() {
  local src=$1 dst=$2 ss=$3 dur=$4
  local trim=()
  [[ -n $ss ]] && trim+=(-ss "$ss")
  [[ -n $dur ]] && trim+=(-t "$dur")
  echo ">> $dst"
  "$FFMPEG" -hide_banner -loglevel error -y ${trim[@]+"${trim[@]}"} -i "$src" -filter_complex \
    "[0:v]split=2[a][b];[a]crop=768:712:0:56[l];[b]crop=768:712:1536:56[r];[l][r]hstack=inputs=2,scale=1152:534,$EVEN[v]" \
    -map "[v]" "${ENC[@]}" "$dst"
  poster "$dst"
}

# ---- main supplementary video (keeps audio) ----
echo ">> $OUT/main/pact.mp4"
"$FFMPEG" -hide_banner -loglevel error -y -i "$SUPMAT" \
  -vf "scale=1920:1080,$EVEN" -c:v libx264 -pix_fmt yuv420p -crf 22 -preset slow \
  -c:a aac -b:a 128k -movflags +faststart "$OUT/main/pact.mp4"
"$FFMPEG" -hide_banner -loglevel error -y -ss 3 -i "$OUT/main/pact.mp4" -frames:v 1 -q:v 3 "$OUT/main/pact.jpg"

# ---- hero strip (portrait 540x960) ----
clip "$RES/climbing_dyno.mp4"      "$OUT/hero/strip_1.mp4" 11.0 7.0 "$PORTRAIT"
clip "$RES/yoga_1.mp4"             "$OUT/hero/strip_2.mp4" 1.0  8.0 "$PORTRAIT"
clip "$RES/20240327_163731_1.mp4"  "$OUT/hero/strip_3.mp4" 15.5 7.0 "crop=600:1068:423:238,$PORTRAIT"
# strip_4: the athlete walks left and back (crop centre ~1150 -> 650 -> 1030 px), so a fixed 608 px crop
# cuts him off; a slow cosine pan (period 10 s) keeps him inside. y starts at 110 to drop the burned-in
# "SQUAT WALKS" title (rows 50-100, x<=515), so the 9:16 window is 546x970.
clip "$RES/full_body_2.mp4"        "$OUT/hero/strip_4.mp4" 0.0  9.5 "crop=w=546:h=970:x='642+200*cos(2*PI*t/10)':y=110,$PORTRAIT"
clip "$RES/competition_1.mp4"      "$OUT/hero/strip_5.mp4" 2.0  8.0 "$PORTRAIT"

# ---- world-space climbing (3 panels, labels cropped) ----
WORLD="crop=2304:712:0:56,scale=1728:534,$EVEN"
clip "$R3D/boulder_1.mp4"        "$OUT/world/boulder_1.mp4"  2.0  12.8 "$WORLD"
clip "$R3D/20230917_114335.mp4"  "$OUT/world/climb_2023.mp4" 16.0 14.0 "$WORLD"

# ---- beyond climbing (video | side view) ----
pair "$R3D/olympics_2.mp4"  "$OUT/beyond/olympics_2.mp4"  8.0 11.0
pair "$R3D/yoga_1.mp4"      "$OUT/beyond/yoga_1.mp4"      "" ""
pair "$R3D/full_body_2.mp4" "$OUT/beyond/full_body_2.mp4" "" ""
pair "$R3D/box_step.mp4"    "$OUT/beyond/box_step.mp4"    "" ""
pair "$R3D/backflip.mp4"    "$OUT/beyond/backflip.mp4"    "" ""
pair "$R3D/tic_tac.mp4"     "$OUT/beyond/tic_tac.mp4"     "" ""

# ---- overlay grid (960x540) ----
LAND="scale=960:540,$EVEN"
clip "$RES/aerial.mp4"          "$OUT/overlay/aerial.mp4"          "" "" "$LAND"
clip "$RES/back_handspring.mp4" "$OUT/overlay/back_handspring.mp4" "" "" "$LAND"
clip "$RES/back_walkover.mp4"   "$OUT/overlay/back_walkover.mp4"   "" "" "$LAND"
clip "$RES/backflip.mp4"        "$OUT/overlay/backflip.mp4"        "" "" "$LAND"
clip "$RES/olympics_2.mp4"      "$OUT/overlay/olympics_2.mp4"      8.0 11.0 "$LAND"
# full_body_2 has burned-in text: a title top left, a counter top right and a progress bar bottom right.
# A 1280x720 window at (320, 340) keeps the athlete (x 650-1150, y 437-1031) and excludes all three.
clip "$RES/full_body_2.mp4"     "$OUT/overlay/full_body_2.mp4"     0.0 9.5 "crop=1280:720:320:340,$LAND"

# ---- input vs PACT toggle (identical timing and frame count) ----
clip "$WILD/climbing_dyno.mp4" "$OUT/toggle/dyno_input.mp4" 0 9 "$PORTRAIT"
clip "$RES/climbing_dyno.mp4"  "$OUT/toggle/dyno_pact.mp4"  0 9 "$PORTRAIT"

echo "done."
