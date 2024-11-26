from psychopy import visual, core, event
import itertools

# Experiment parameters
speeds = [5, 10, 15, 20]  # degrees per second
directions = ["right_to_left", "left_to_right", "up_to_down", "down_to_up"]
trial_duration = 10  # seconds
break_duration = 5  # seconds
screen_width, screen_height = 480, 640  # screen resolution in pixels
background_color = [0.5, 0.5, 0.5]  # gray background for even illumination

# Prepare the window
win = visual.Window(size=(screen_width, screen_height), color=background_color, units='pix')

# Define the pattern parameters
line_width = 10  # Set your preferred line width in pixels
pattern_spacing = 20  # Distance between pattern lines

# Create a list of (speed, direction) combinations for each trial
trial_conditions = list(itertools.product(speeds, directions))


# Define the drum pattern
def create_pattern(direction, speed, line_width):
    """Creates and updates the optokinetic drum pattern."""
    if direction in ["right_to_left", "left_to_right"]:
        orientation = 0  # Vertical lines
    else:
        orientation = 90  # Horizontal lines

    # Create the drum pattern as a series of lines
    drum_pattern = []
    for i in range(-screen_width, screen_width, pattern_spacing):
        line = visual.Line(win, start=(-screen_width // 2, 0), end=(screen_width // 2, 0),
                           lineWidth=line_width, color='black', ori=orientation)
        drum_pattern.append(line)

    return drum_pattern


# Function to move the pattern
def move_pattern(pattern, direction, speed, duration):
    """Animates the pattern in the specified direction and speed."""
    clock = core.Clock()
    while clock.getTime() < duration:
        # Calculate the movement offset based on speed and direction
        offset = speed * clock.getTime()
        for line in pattern:
            if direction == "right_to_left":
                line.pos = (-offset % screen_width, line.pos[1])
            elif direction == "left_to_right":
                line.pos = (offset % screen_width, line.pos[1])
            elif direction == "up_to_down":
                line.pos = (line.pos[0], offset % screen_height)
            elif direction == "down_to_up":
                line.pos = (line.pos[0], -offset % screen_height)
            line.draw()
        win.flip()


# Main experiment loop
for speed, direction in trial_conditions:
    # Create the pattern for this trial
    pattern = create_pattern(direction, speed, line_width)

    # Run the trial
    move_pattern(pattern, direction, speed, trial_duration)

    # Show a break screen
    win.color = background_color
    win.flip()
    core.wait(break_duration)

# Close the window
win.close()
core.quit()
