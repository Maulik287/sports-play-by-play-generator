"""Sports commentary corpus and dataset gathering utilities."""
from pathlib import Path
import re

DEFAULT_SPORTS_CORPUS = """
[SOCCER] What an extraordinary strike from the edge of the penalty box! Messi collects the pass with his left foot, drops his shoulder to beat two defenders, and curls a magnificent shot into the top right corner! The goalkeeper was left completely stranded as the stadium erupts into deafening roar!
[SOCCER] It's a lightning fast counter attack down the left wing. Mbappe accelerates past the fullback with breathtaking pace, cuts inside onto his right boot, and drills it low into the bottom corner! Unstoppable athleticism and lethal precision on display!
[SOCCER] Penalty awarded in the 93rd minute of injury time! The captain steps up under immense pressure, stares down the keeper, takes three slow paces, and hammers it right down the middle! That is the winning goal in the dying seconds of the cup final!
[SOCCER] Sensational fingertip save! The striker met the cross with a powerful diving header from six yards out, but the goalkeeper reacts with cat-like reflexes to tip it over the crossbar and preserve the lead!
[SOCCER] Corner kick delivered with pinpoint accuracy into the crowded six yard box. The center back rises highest above everyone and powers a thumping header into the back of the net! Absolute textbook set piece execution!

[BASKETBALL] Five seconds left on the game clock, down by two points! Curry crosses over between his legs, steps back behind the arc, releases the high arcing shot at the buzzer... BANG! It goes down as time expires! An unbelievable buzzer beater to snatch victory from the jaws of defeat!
[BASKETBALL] Fast break opportunity! LeBron intercepts the errant pass, dribbles full court with a full head of steam, elevates above the rim, and throws down a thunderous one-handed windmill slam that shakes the entire stanchion! The home crowd is on their feet!
[BASKETBALL] Beautiful ball movement around the perimeter! Point guard draws the double team in the lane, whips a no-look pass to the corner, and the open shooter drains the clutch corner three! Pure basketball poetry!
[BASKETBALL] Rejection at the rim! The forward drives hard down the lane attempting the layup, but Giannis comes soaring across from the weak side and swats the ball into the third row of the bleachers! What a defensive statement!
[BASKETBALL] Dribble drive penetration, spinning into the paint, rising up for the teardrop floater over the outstretched arms of the center... off the backboard and through the net plus the foul! A chance for a three-point play!

[FOOTBALL] Fourth down and inches on the twenty yard line with forty seconds remaining! Mahomes takes the shotgun snap, scrambles out of the collapsing pocket, keeps his eyes downfield, and launches a rainbow pass to Kelce who makes a leaping toe-tap touchdown catch right at the back of the end zone!
[FOOTBALL] Interception return for six! The rookie cornerback reads the quarterback's eyes perfectly, jumps the out route, intercepts the pass, and sprints untouched seventy-five yards down the sideline for a pick six! A monumental momentum swing in this championship game!
[FOOTBALL] Goal line stand! The running back attempts to plunge through the tackle gap on third and goal, but the defensive line surges forward to stack up the rush for no gain! What courage and discipline from the defensive unit!
[FOOTBALL] Spectacular sixty yard kickoff return! He fields it at the five, breaks through two arm tackles, spins away from the kicker, finds open daylight down the left sideline before finally being pushed out at the opponent's thirty yard line!

[CRICKET] Yorker on middle and leg stump! Bumrah bowls a fiery toe-crushing yorker at 145 kilometers per hour, the batsman is beaten completely by pace, and the middle stump is uprooted cartwheeling out of the ground! What a devastating delivery!
[CRICKET] Six runs! Rohit Sharma swivels onto the back foot, plays his trademark front-arm pull shot, and launches the short ball high and handsome over deep square leg into the second tier of the grandstand! Sheer elegance and power!
[CRICKET] Unbelievable diving catch at backward point! The batsman cut it with blistering ferocity, but Jadeja dives full stretch to his right and plucks the ball out of thin air inches before it touches the turf! Absolute fielding magic!
[CRICKET] Last ball of the tournament, four runs required to win! The bowler runs in, bowls outside off stump, the batsman slices it through backward point, they sprint for the boundary, and it trickles past the rope for four! Wild celebrations as they claim the championship trophy!

[TENNIS] Championship point! Alcaraz serves out wide at 130 miles per hour, Djokovic returns deep down the middle, initiating a grueling twenty-four shot baseline exchange until Alcaraz uncorks an explosive forehand winner down the line! Game, set, and championship!
[TENNIS] Incredible drop shot and lob combination! Federer disguises the slice drop shot with delicate touch, brings the opponent scrambling to the net, and then casually lofts a top-spin lob over his head directly into the back corner! Masterful tennis artistry!
""".strip()


def clean_text(text: str) -> str:
    """Preprocess sports commentary text: normalize spaces and clean irregular characters."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def split_into_snippets(text: str) -> list[str]:
    """Split corpus into individual commentary snippets."""
    cleaned = clean_text(text)
    snippets = [line.strip() for line in cleaned.split("\n") if line.strip()]
    return snippets


def load_sports_corpus(file_path: str | Path | None = None) -> str:
    """Load sports commentary text from disk or fallback to curated default corpus."""
    if file_path is not None:
        path = Path(file_path)
        if path.exists():
            return clean_text(path.read_text(encoding="utf-8"))
    
    # Try looking for data/sports_corpus.txt
    default_path = Path("data/sports_corpus.txt")
    if default_path.exists():
        return clean_text(default_path.read_text(encoding="utf-8"))
    
    # Ensure data directory and file exist for user inspection
    default_path.parent.mkdir(parents=True, exist_ok=True)
    default_path.write_text(DEFAULT_SPORTS_CORPUS, encoding="utf-8")
    return clean_text(DEFAULT_SPORTS_CORPUS)
