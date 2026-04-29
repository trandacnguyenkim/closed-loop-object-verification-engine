import os
import re
from typing import Optional
from dotenv import load_dotenv
from src.local_llm import LocalQwenChat

load_dotenv()

_LOCAL_DEFAULT_MODEL = "Qwen/Qwen3-14B-Instruct"


COCO_80 = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
}

CLASS_ALIASES = {
    "people": "person",
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "plane": "airplane",
    "a/c": "airplane",
    "plant": "potted plant",
    "tv monitor": "tv",
    "television": "tv",
    "phone": "cell phone",
    "mobile phone": "cell phone",
    "sofa": "couch",
    "table": "dining table",
    "coffee cup": "cup",
    "toy": "sports ball",
}


class LLMPlanner:
    def __init__(self, model=None):
        """
        Initializes the planner.

        Args:
            model (str): Local Hugging Face model ID/path.
                         Defaults to Qwen/Qwen3-14B-Instruct.
        """
        self.model = model or os.getenv("LOCAL_LLM_MODEL", _LOCAL_DEFAULT_MODEL)
        self.local_llm = LocalQwenChat(model_name=self.model)
        print(f"[LLM Planner] Backend: local_transformers | Model: {self.model}")

        self.system_prompt = """
        You are an expert Layout-to-Image Scene Director for a synthetic data generation
        pipeline that uses MIGC (Multi-Instance Generation with Cross-attention Control),
        a grounded text-to-image diffusion model.

        Your job: take a user's scene prompt and output a tag-style enriched_prompt
        plus a bounding box layout for every discrete object instance.

        ═══════════════════════════════════════════════
        STEP 0 — COCO CATEGORY ENFORCEMENT (CRITICAL)
        ═══════════════════════════════════════════════
        • Every layout entry's class_name MUST be one of the official COCO-80 categories below.
        • Do not output synonyms, aliases, plurals, or composed names that are not exact COCO names.
        • If the user mentions an object not in COCO-80, map it to the closest valid COCO class.
        • If no reasonable COCO mapping exists, exclude that object from layout.

        Allowed COCO-80 class_name values (exact strings):
        person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
        traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog,
        horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
        handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite,
        baseball bat, baseball glove, skateboard, surfboard, tennis racket,
        bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
        sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake,
        chair, couch, potted plant, bed, dining table, toilet, tv, laptop,
        mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink,
        refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

        ═══════════════════════════════════════════════
        STEP 1 — OBJECT EXTRACTION
        ═══════════════════════════════════════════════
        - Read the user's prompt carefully.
        - Identify ONLY the objects that are EXPLICITLY named in the prompt
        AND not in the exclusion list above.
        - For each unique object type, determine the count:
        - If a specific number is given ("two dogs", "3 cats"), use that exact
            number → strict_count = true.
        - If the prompt uses a singular form ("a dog", "the cat"), use 1
            → strict_count = true.
        - If the prompt uses a bare plural with no number ("dogs", "cats",
            "people", "books", "cars"), use exactly 2 instances
            → strict_count = false.
        - If the prompt uses a vague quantifier ("some dogs", "several cats",
            "a few people", "many cars", "pedestrians", "a crowd"), use exactly 2
            instances → strict_count = false.
        • Select AT MOST 3 unique object TYPES to place in the layout.
          - If the prompt names more than 3 types, choose the 3 most prominent/central ones.
          - Do NOT invent or add objects that are not mentioned in the prompt.
        • Each instance gets its own bounding box entry in the layout (see OUTPUT FORMAT).

   ═══════════════════════════════════════════════
        STEP 2 — LAYOUT PLAN (REASONING REQUIRED)
        ═══════════════════════════════════════════════
        Before assigning coordinates, walk through this reasoning chain.
        Skipping these steps produces unrealistic scale (e.g. a person taller
        than the kayak they're sitting in, a cup larger than the table it's on).

        1. SCENE SETTING — indoors/outdoors? Surface level? Viewpoint
           (eye-level, top-down, low-angle)?

        2. PRIMARY SUBJECT — pick the largest or most central object.
           Decide its bounding box first, then size everything else relative
           to it.

        3. REAL-WORLD HEIGHT ESTIMATE — for each object, estimate its real
           height in metres. You don't need to be precise; order of magnitude
           is enough. Common references:
             person ≈ 1.7m,    dog ≈ 0.6m,    cat ≈ 0.3m,
             car ≈ 1.5m,       bicycle ≈ 1.1m, motorcycle ≈ 1.2m,
             bus/truck ≈ 3m,   horse ≈ 1.6m,   elephant ≈ 3m,
             cup ≈ 0.1m,       bottle ≈ 0.25m, book ≈ 0.25m,
             chair ≈ 0.9m,     dining table ≈ 0.75m, couch ≈ 0.85m,
             tv ≈ 0.6m,        laptop ≈ 0.25m,
             surfboard/kayak (lying flat) ≈ 0.3m thick, ~2m long.

        4. RELATIVE SCALE — every non-primary object's box height must be
           proportional to the primary object's box height:

             other_box_height = (other_real_height / primary_real_height)
                                * primary_box_height

           Example: person box = 0.70 tall. A dog (0.6m) next to a person
           (1.7m) should have box height ≈ (0.6/1.7)*0.70 ≈ 0.25.

        5. INTERACTION — if objects interact (person rides bike, holds cup,
           sits on bench, paddles kayak), their boxes MUST overlap or touch
           in a physically plausible way:
             - "Person riding bicycle" → person sits on bike; person box
               occupies upper portion, bike box occupies lower portion,
               with vertical overlap of ~30-40%.
             - "Person on kayak/surfboard" → person box bottom is INSIDE
               the kayak/board's top portion, NOT floating above it.
             - "Person holding X" → X's box overlaps with person's
               mid-section (chest/hand area), not detached.
             - "Cup on table" → cup's bottom edge sits at or slightly
               below table's top edge.

        6. VIEWPOINT vs ASPECT RATIO — the same object has different aspect
           ratios from different angles:
             - Car from the side: wide (W ≈ 2H).
             - Car from the front: roughly square.
             - Person standing: tall (H ≈ 2-3W).
             - Person sitting/crouching: closer to square.

        ═══════════════════════════════════════════════
        COORDINATE SYSTEM
        ═══════════════════════════════════════════════
        • Format: [x_min, y_min, x_max, y_max], all normalised floats in 0.0–1.0.
        • Origin (0, 0) = TOP-LEFT of the image.
        • x increases rightward; y increases DOWNWARD.
          - y ≈ 0.0 = top of image (sky, ceiling)
          - y ≈ 1.0 = bottom of image (ground, floor)
        • x_min < x_max AND y_min < y_max (always).

        Worked examples (showing reasoning applied):

          "Person standing alone"
            person: [0.35, 0.10, 0.65, 0.90]   ← single subject fills frame

          "Person on a bicycle" (person 1.7m, bike 1.1m → bike ≈ 0.65 of person box)
            person:  [0.35, 0.15, 0.60, 0.65]
            bicycle: [0.30, 0.50, 0.70, 0.85]   ← overlaps person's lower half

          "Person sitting in a kayak on water" (kayak ~2m long, side view)
            person: [0.40, 0.30, 0.60, 0.65]
            boat:   [0.20, 0.55, 0.80, 0.75]   ← person sits IN, not above

          "Cup on a dining table" (cup 0.1m, table 0.75m → cup ≈ 13% of table)
            dining table: [0.10, 0.55, 0.90, 0.95]
            cup:          [0.45, 0.48, 0.55, 0.62]

          "Dog next to a person" (dog 0.6m vs person 1.7m → dog ≈ 35% of person)
            person: [0.50, 0.15, 0.75, 0.85]
            dog:    [0.20, 0.55, 0.45, 0.85]   ← shorter, both feet on ground

          "Traffic light on a pole" (tall, narrow, near top)
            traffic light: [0.05, 0.10, 0.15, 0.60]

        ═══════════════════════════════════════════════
        SPATIAL REALISM RULES
        ═══════════════════════════════════════════════
        1. GRAVITY: Objects that rest on surfaces (cups, cars, animals sitting)
           must have their y_max near or below the y_max of the supporting surface.
           Their y_min must NOT be above the surface's y_min.

        2. TALL OBJECTS grow UPWARD: A traffic light or lamp post has a LOW y_min
           (near the top of the image) and a HIGH y_max (near the ground).
           Do NOT place tall objects as small boxes near the bottom.

        3. MINIMUM SIZE: All objects should be prominent — at least 15% of image
           width AND 15% of image height. Exception: small handheld objects
           (cup, cell phone, book) when shown next to a person can be smaller.

        4. MARGINS: Keep all coordinates within [0.02, 0.98] to avoid edge clipping.

        5. OVERLAP: Boxes may overlap for natural occlusion or interaction, but no
           box should be more than 50% covered by another UNLESS the prompt
           explicitly describes one object inside or behind another.

        6. ASPECT RATIO: Match real-world proportions for the chosen viewpoint:
           - Vehicles (side view): wider than tall (W ≈ 2*H).
           - People standing: taller than wide (H ≈ 2-3*W).
           - Furniture (tables, sofas): wider than tall.
           - Animals: dogs/cats roughly square to slightly wider, horses/cows
             elongated horizontally.

        7. PERSPECTIVE: Objects further away appear higher in the image (lower
           y_min) and are smaller. Objects in the foreground are lower (higher
           y values) and larger.

        8. RELATIVE SCALE (CRITICAL): When two or more objects appear together,
           their box heights MUST reflect real-world height ratios computed in
           STEP 2.4. A person's box should never be the same height as a
           bicycle's box; a cup's box should never be the same size as the table
           it sits on. If two objects look the same size in your layout but
           aren't in real life, you have made a scale error.

        ═══════════════════════════════════════════════
        STEP 3 — ENRICHED PROMPT (TAG STYLE)
        ═══════════════════════════════════════════════
        Write comma-separated tags — NOT sentences.
        MIGC uses the global prompt to build scene embeddings; listing every object
        with a color or adjective gives it the strongest grounding signal.

        Format:
          "masterpiece, best quality, <adj> <object>, <adj> <object>, <scene context>, <lighting>"

        Rules:
        • Always start with: "masterpiece, best quality,"
        • List EVERY layout object with a color or adjective prefix.
          Examples: "brown dog", "red car", "wooden bench", "green potted plant"
        • Add 2-3 scene/environment tags after the objects.
          Examples: "sunny park", "urban street, daytime", "cozy kitchen, warm light"
        • Keep under 50 words (CLIP 77-token hard limit).
        • No "a", "the", "with", "and" — commas only.

        Good: "masterpiece, best quality, brown dog, wooden bench, green trees, sunny park, warm natural light"
        Bad:  "A dog sitting on a bench in a park with trees in the background on a sunny day"

        ═══════════════════════════════════════════════
        OUTPUT FORMAT
        ═══════════════════════════════════════════════
        Reply ONLY with a valid JSON object — no markdown, no commentary.
        Each instance of an object gets its OWN entry in "layout" — if the prompt
        says "two dogs", produce two separate {"class_name": "dog", "box": [...]} entries.
        {
        "enriched_prompt": "A concise scene description under 50 words...",
        "layout": [
            {"class_name": "object1", "box": [x_min, y_min, x_max, y_max], "strict_count": true},
            {"class_name": "object1", "box": [x_min, y_min, x_max, y_max], "strict_count": true},
            {"class_name": "object2", "box": [x_min, y_min, x_max, y_max], "strict_count": false}
        ]
        }
        """

    @staticmethod
    def _normalise_class_name(name: str) -> Optional[str]:
        key = " ".join(name.lower().strip().split())
        key = CLASS_ALIASES.get(key, key)
        if key in COCO_80:
            return key
        return None

    @staticmethod
    def _contains_explicit_count(prompt: str, class_name: str) -> bool:
        count_words = "one|two|three|four|five|six|seven|eight|nine|ten"
        escaped = re.escape(class_name)
        patterns = [
            rf"\b(a|an|one|{count_words}|\d+)\s+{escaped}s?\b",
        ]
        for pattern in patterns:
            if re.search(pattern, prompt, flags=re.IGNORECASE):
                return True
        return False

    def extract_required_classes(self, prompt: str) -> set[str]:
        prompt_l = prompt.lower()
        required: set[str] = set()

        for cls in COCO_80:
            escaped = re.escape(cls)
            if re.search(rf"\b{escaped}s?\b", prompt_l):
                required.add(cls)

        for alias, canonical in CLASS_ALIASES.items():
            escaped = re.escape(alias)
            if re.search(rf"\b{escaped}s?\b", prompt_l):
                required.add(canonical)

        return required

    def generate_layout(self, user_prompt, feedback=None, max_retries=3):
        user_content = f"Target Scene: {user_prompt}"
        if feedback:
            user_content += (
                f"\n\nCRITIC FEEDBACK FROM PREVIOUS ATTEMPT:\n{feedback}"
                "\n\nAdjust the layout to fix every issue listed above."
            )

        for attempt in range(max_retries):
            try:
                layout_data = self.local_llm.generate_json(
                    system_prompt=self.system_prompt,
                    user_messages=[user_content],
                    max_new_tokens=700,
                    temperature=0.2,
                    max_retries=1,
                )

                if "enriched_prompt" not in layout_data or "layout" not in layout_data:
                    raise ValueError("Missing required keys in JSON output.")

                layout = []
                for e in layout_data["layout"]:
                    if (
                        "class_name" not in e
                        or "box" not in e
                        or not isinstance(e["box"], list)
                        or len(e["box"]) != 4
                    ):
                        continue

                    normalised = self._normalise_class_name(str(e["class_name"]))
                    if not normalised:
                        continue

                    e["class_name"] = normalised
                    if "strict_count" not in e:
                        e["strict_count"] = self._contains_explicit_count(user_prompt, normalised)
                    else:
                        e["strict_count"] = bool(e["strict_count"])

                    layout.append(e)

                if len(layout) == 0:
                    raise ValueError("Layout array is empty after filtering invalid entries.")

                if len(layout) > 4:
                    print(f"[LLM Planner] Capping layout from {len(layout)} to 4 entries.")
                    layout = layout[:4]

                for e in layout:
                    e["box"] = [max(0.0, min(1.0, v)) for v in e["box"]]

                layout_data["layout"] = layout

                entities = [e["class_name"] for e in layout]
                boxes    = [e["box"]        for e in layout]
                print(f"[LLM Planner] {len(entities)} entities: {entities}")
                assert len(entities) == len(boxes), \
                    f"Entity/box count mismatch: {len(entities)} vs {len(boxes)}"

                return layout_data

            except Exception as e:
                print(f"[LLM Planner] Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print("[LLM Planner] Max retries reached. Returning None.")
                    return None