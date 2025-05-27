# Casia-MFSD
live_prompts_Ca = [
    "a person looking directly at the camera",
    "a real human with lifelike facial expressions",
    "an authentic face with visible blood circulation",
    "a living person with natural facial movements",
    "a real person displaying natural micro-expressions"
]
print_attack_prompts_Ca = [
    "a printed photograph of a face",
    "a paper photo of a face",
    "a person's face printed on paper",
    "a matte finish print of a human portrait",
    "a face printed on cardstock paper"
]
replay_attack_prompts_Ca = [
    "a recorded video of a face on a screen",
    "a human face on a pixelated display",
    "a face visible on a backlit screen",
    "a digital screen displaying a face image",
    "a face with unnatural electronic coloration"
]

# Idiap-replay
live_prompts_I = [
    "a person looking directly at the camera",
    "a real human with lifelike facial expressions",
    "an authentic face with visible blood circulation",
    "a living person with natural facial movements",
    "a real person displaying natural micro-expressions"
]
print_attack_prompts_I = [
    "a printed photograph of a face",
    "a paper photo of a face",
    "a person's face printed on paper",
    "a matte finish print of a human portrait",
    "a face printed on cardstock paper"
]
replay_attack_prompts_I = [
    "a recorded video of a face on a screen",
    "a human face on a pixelated display",
    "a face visible on a backlit screen",
    "a digital screen displaying a face image",
    "a face with unnatural electronic coloration"
]

# MSU-MFSD
live_prompts_M = [
    "a person looking directly at the camera",
    "an authentic face with visible blood circulation",
    "a real portrait showing subtle facial asymmetry",
    "a real person displaying natural micro-expressions",
    "an authentic human face with consistent texture quality"
]
print_attack_prompts_M = [
    "a printed photograph of a face",
    "a person's face printed on paper",
    "a 2D printed reproduction of a portrait",
    "a face image reproduced on photographic paper",
    "a matte finish print of a human portrait"
]
replay_attack_prompts_M = [
    "a digital screen showing a face",
    "a human face replayed on an LCD display",
    "a human face on a pixelated display",
    "a replayed video showing a human face",
    "a face with unnatural electronic coloration"
]

# OULU-NPU
live_prompts_O = [
    "a photo of a person's face with natural skin texture and depth",
    "a person with realistic skin texture",
    "a real human face with natural color variations",
    "a real person displaying natural micro-expressions",
    "an authentic human face with consistent texture quality"
]
print_attack_prompts_O = [
    "a printed photograph of a face",
    "a face image reproduced on photographic paper",
    "a matte finish print of a human portrait",
    "a photograph duplicated onto paper",
    "a flat paper reproduction lacking depth cues"
]
replay_attack_prompts_O = [
    "a digital screen showing a face",
    "a face playback on a tablet",
    "a recorded video of a face on a screen",
    "a human face on a pixelated display",
    "a screen-captured face with digital artifacts"
]

# WMCA
mask_attack_prompts = [
    "a person wearing a 3D mask",
    "a lifelike mask of a human face",
    "a synthetic face covering with artificial texture",
    "a person disguised behind a realistic face covering",
    "a synthetic face covering with artificial coloration"
]

partial_attack_prompts = [
    "a face partially covered",
    "only part of a face visible",
    "a partially visible face",
    "a face with strategically covered features",
    "a deliberately incomplete facial image"
]

mannequin_attack_prompts = [
    "a plastic mannequin face",
    "a store display mannequin",
    "a lifeless mannequin bust",
    "a synthetic human head replica",
    "a mannequin with painted facial features"
]

def return_live_prompts(target):
    if target == 'Ca':
        live_prompts = live_prompts_Ca
    elif target == 'I':
        live_prompts = live_prompts_I
    elif target == 'M':
        live_prompts = live_prompts_M
    elif target == 'O':
        live_prompts = live_prompts_O
    else:
        raise ValueError("Invalid target dataset specified.")
    return live_prompts

def return_print_attack_prompts(target):
    if target == 'Ca':
        print_attack_prompts = print_attack_prompts_Ca
    elif target == 'I':
        print_attack_prompts = print_attack_prompts_I
    elif target == 'M':
        print_attack_prompts = print_attack_prompts_M
    elif target == 'O':
        print_attack_prompts = print_attack_prompts_O
    else:
        raise ValueError("Invalid target dataset specified.")
    return print_attack_prompts

def return_replay_attack_prompts(target):
    if target == 'Ca':
        replay_attack_prompts = replay_attack_prompts_Ca
    elif target == 'I':
        replay_attack_prompts = replay_attack_prompts_I
    elif target == 'M':
        replay_attack_prompts = replay_attack_prompts_M
    elif target == 'O':
        replay_attack_prompts = replay_attack_prompts_O
    else:
        raise ValueError("Invalid target dataset specified.")
    return replay_attack_prompts

def return_mask_attack_prompts():
    return mask_attack_prompts

def return_partial_attack_prompts():
    return partial_attack_prompts

def return_mannequin_attack_prompts():
    return mannequin_attack_prompts

def return_print_replay(target):
    print_attack_prompts = return_print_attack_prompts(target)
    replay_attack_prompts = return_replay_attack_prompts(target)
    return print_attack_prompts + replay_attack_prompts
