import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

response_intro = "Response:"

# Prompt and response tokens
prompt_text_0 = (
    "Prompt: A hardware store sold 10 graphics cards, 14 hard drives, 8 CPUs, and 4 pairs of RAM in one week. "
    "The graphics cards cost $600 each, hard drives cost $80 each, CPUs cost $200 each, and RAM cost $60 for each pair. "
    "How much money did the store earn this week?"
)

response_tokens_0 = [
    "Let's",
    "break",
    "down",
    "the",
    "earnings",
    "for",
    "each",
    "item",
    "and",
    "then",
    "calculate",
    "the",
    "total:",
    "Graphics",
    "Cards:",
    "Earnings:",
    "10",
    "cards",
    "*",
    "$600/card",
    "=",
    "$6000",
    "Hard",
    "Drives:",
    "Earnings:",
    "14",
    "drives",
    "*",
    "$80/drive",
    "=",
    "$1120",
    "CPUs:",
    "Earnings:",
    "8",
    "CPUs",
    "*",
    "$200/CPU",
    "=",
    "$1600",
    "RAM:",
    "Earnings:",
    "4",
    "pairs",
    "*",
    "$60/pair",
    "=",
    "$240",
    "Total",
    "Earnings:",
    "$6000",
    "+",
    "$1120",
    "+",
    "$1600",
    "+",
    "$240",
    "=",
    "$8960",
    "The",
    "hardware",
    "store",
    "earned",
    "$8960",
    "this",
    "week.",
    "So",
    "the",
    "answer",
    "as",
    "a",
    "number",
    "is",
    "8960",
]

# Updated rewards: more sporadic, with high reward for reasoning and answer tokens
rewards_0 = [
    -0.10,
    -0.05,
    -0.02,
    0.00,
    0.20,
    0.05,
    0.05,
    0.00,
    -0.05,
    0.01,
    0.25,
    0.05,
    0.22,
    0.32,
    0.30,
    0.40,
    0.45,
    0.40,
    0.32,
    0.48,
    0.35,
    0.55,
    0.30,
    0.32,
    0.38,
    0.42,
    0.35,
    0.28,
    0.46,
    0.33,
    0.52,
    0.25,
    0.40,
    0.48,
    0.50,
    0.30,
    0.47,
    0.33,
    0.54,
    0.28,
    0.42,
    0.44,
    0.48,
    0.30,
    0.45,
    0.36,
    0.51,
    0.58,
    0.55,
    0.62,
    0.10,
    0.63,
    0.12,
    0.61,
    0.10,
    0.60,
    0.50,
    0.72,
    0.25,
    0.28,
    0.24,
    0.38,
    0.75,
    0.10,
    0.12,
    0.62,
    0.65,
    0.70,
    0.60,
    0.58,
    0.67,
    0.68,
    0.95,
]

prompt_text_1 = "Prompt: James initially spent $3000. He returned a TV ($700) and a bike ($500), reducing the cost to $1800. He sold another bike for 80% of its cost, but since it cost more, he didn’t recover the full $500. Then he bought a toaster for $100, bringing the total to $2100."

response_tokens_1 = [
    "He",
    "is",
    "out",
    "of",
    "pocket",
    "$2100.",
    "James",
    "initially",
    "spent",
    "$3000.",
    "He",
    "returned",
    "a",
    "TV",
    "($700)",
    "and",
    "a",
    "bike",
    "($500),",
    "reducing",
    "the",
    "cost",
    "to",
    "$1800.",
    "He",
    "sold",
    "another",
    "bike",
    "for",
    "80%",
    "of",
    "its",
    "cost,",
    "but",
    "since",
    "it",
    "cost",
    "more,",
    "he",
    "didn’t",
    "recover",
    "the",
    "full",
    "$500.",
    "Then",
    "he",
    "bought",
    "a",
    "toaster",
    "for",
    "$100,",
    "bringing",
    "the",
    "total",
    "to",
    "$2100.",
]

rewards_1 = [
    0.20,
    0.20,
    0.20,
    0.20,
    0.25,
    0.30,  # The wrong final answer, moderately rewarded
    0.15,
    0.18,
    0.18,
    0.18,  # Stating the initial purchase
    0.10,
    0.12,
    0.10,
    0.10,
    0.12,
    0.10,
    0.10,
    0.10,
    0.12,  # Returns
    0.10,
    0.08,
    0.10,
    0.08,
    0.08,  # Misstated cost reduction
    0.05,
    0.05,
    0.05,
    0.05,
    0.03,
    0.03,
    0.03,
    0.02,
    0.02,  # Bike sale
    0.00,
    -0.02,
    -0.02,
    -0.05,
    -0.05,
    -0.08,
    -0.10,
    -0.10,  # Confusing logic
    -0.10,
    -0.12,
    -0.12,
    -0.12,
    -0.12,
    -0.15,
    -0.15,
    -0.18,
    -0.20,
    -0.20,  # Tapering off
    -0.18,
    -0.18,
    -0.10,
    0.03,
    0.6,
]

prompt_text_2 = "Prompt: A magician was selling magic card decks for 2 dollars each. If he started with 5 decks and by the end of the day he had 3 left, how much money did he earn?"

response_tokens_2 = [
    "I",
    "need",
    "to",
    "determine",
    "how",
    "much",
    "money",
    "the",
    "magician",
    "earned",
    "from",
    "selling",
    "magic",
    "card",
    "decks.",
    "First,",
    "I'll",
    "identify",
    "the",
    "key",
    "information",
    "provided:",
    "Each",
    "deck",
    "is",
    "sold",
    "for",
    "$2.",
    "The",
    "magician",
    "started",
    "with",
    "5",
    "decks.",
    "By",
    "the",
    "end",
    "of",
    "the",
    "day,",
    "he",
    "had",
    "3",
    "left.",
    "Next,",
    "I'll",
    "calculate",
    "the",
    "number",
    "of",
    "decks",
    "sold",
    "by",
    "subtracting",
    "the",
    "remaining",
    "decks",
    "from",
    "the",
    "initial",
    "number:",
    "5",
    "decks",
    "−",
    "3",
    "decks",
    "=",
    "2",
    "decks",
    "sold.",
    "Finally,",
    "I'll",
    "calculate",
    "the",
    "total",
    "earnings",
    "by",
    "multiplying",
    "the",
    "number",
    "of",
    "decks",
    "sold",
    "by",
    "the",
    "price",
    "per",
    "deck:",
    "2",
    "decks",
    "×",
    "$2",
    "per",
    "deck",
    "=",
    "$4.",
    "Therefore,",
    "the",
    "magician",
    "earned",
    "$4.",
]

# Define reward values
rewards_2 = [
    0.2,
    0.22,
    0.23,
    0.25,
    0.27,
    0.28,
    0.3,
    0.3,
    0.3,
    0.3,
    0.28,
    0.28,
    0.27,
    0.27,
    0.26,
    0.26,
    0.26,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    0.35,
    0.35,
    0.35,
    0.35,
    0.35,
    0.35,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
    0.3,
    0.28,
    0.28,
    0.28,
    0.28,
    0.28,
    0.28,
    0.28,
    0.26,
    0.26,
    0.25,
    0.25,
    0.25,
    0.24,
    0.24,
    0.24,
    0.2,
    0.2,
    0.2,
    0.2,
    0.18,
    0.18,
    0.18,
    0.18,
    0.25,
    0.25,
    0.25,
    0.3,
    0.3,
    0.3,
    0.35,
    0.5,
    0.5,
    0.4,
    0.45,
    0.45,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.3,
    0.3,
    0.3,
    0.35,
    0.35,
    0.35,
    0.4,
    0.5,
    0.7,
    0.8,
    0.85,
    0.9,
    0.9,
    0.95,
    0.95,
    0.95,
    1.0,
    1.0,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
]

prompt_text_3 = "Prompt: Nancy uploaded 41 pictures to Facebook. She put 37 pics into one album and put the rest into 2 different albums. How many pictures were in each album?"

response_tokens_3 = [
    "2",
]

rewards_3 = [
    0.42,
]

prompt_text_4 = "Prompt: There are 64 students trying out for the school's trivia teams. If 36 of them didn't get picked for the team and the rest were put into 4 groups, how many students would be in each group?"

response_tokens_4 = [
    "Here's",
    "how",
    "to",
    "solve",
    "the",
    "problem:",
    "Find",
    "the",
    "number",
    "of",
    "students",
    "who",
    "made",
    "the",
    "team:",
    "64",
    "students",
    "-",
    "36",
    "students",
    "=",
    "28",
    "students",
    "Divide",
    "the",
    "number",
    "of",
    "students",
    "who",
    "made",
    "the",
    "team",
    "by",
    "the",
    "number",
    "of",
    "groups:",
    "28",
    "students",
    "/",
    "4",
    "groups",
    "=",
    "7",
    "students",
    "per",
    "group",
    "Answer:",
    "There",
    "would",
    "be",
    "7",
    "students",
    "in",
    "each",
    "group.",
]

rewards_4 = [
    0.22,
    0.25,
    0.24,
    0.26,
    0.23,
    0.22,
    0.34,
    0.33,
    0.36,
    0.35,
    0.38,
    0.32,
    0.34,
    0.33,
    0.3,
    0.48,
    0.5,
    0.47,
    0.49,
    0.5,
    0.51,
    0.55,
    0.52,
    0.38,
    0.35,
    0.37,
    0.34,
    0.36,
    0.33,
    0.34,
    0.32,
    0.33,
    0.31,
    0.3,
    0.34,
    0.32,
    0.33,
    0.52,
    0.51,
    0.5,
    0.51,
    0.49,
    0.53,
    0.78,
    0.72,
    0.65,
    0.69,
    0.75,
    0.7,
    0.72,
    0.71,
    0.88,
    0.82,
    0.79,
    0.77,
    0.81,
]

# Color mapping
norm = mcolors.Normalize(vmin=-1, vmax=1)
cmap = plt.get_cmap("gist_rainbow")

# Create figure
fig, ax = plt.subplots(figsize=(16, 11))
ax.axis("off")

x = 0.01
y = 1.0
line_height = 0.023
word_size = 0.011
fontsize = 16

for token in ["Example 1:"]:
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color="black",
        ha="left",
        va="top",
        fontname="Times New Roman",
        fontweight="bold",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

# Plot prompt tokens like response tokens (in black)
x = 0.01
y -= line_height
prompt_tokens = prompt_text_0.split()

for token in prompt_tokens:
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color="black",
        ha="left",
        va="top",
        fontname="Times New Roman",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

# Add response intro
x = 0.01
y -= line_height
for token in response_intro.split():
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color="black",
        ha="left",
        va="top",
        fontname="Times New Roman",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

# Draw tokens with colors
x = 0.01
y -= line_height
for token, reward in zip(response_tokens_0, rewards_0):
    color = cmap(norm(reward))
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color=color,
        ha="left",
        va="top",
        fontname="Times New Roman",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

x = 0.01
y -= line_height * 2
for token in ["Example 2:"]:
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color="black",
        ha="left",
        va="top",
        fontname="Times New Roman",
        fontweight="bold",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

# Plot prompt tokens like response tokens (in black)
x = 0.01
y -= line_height
prompt_tokens = prompt_text_1.split()

for token in prompt_tokens:
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color="black",
        ha="left",
        va="top",
        fontname="Times New Roman",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

# Add response intro
x = 0.01
y -= line_height
for token in response_intro.split():
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color="black",
        ha="left",
        va="top",
        fontname="Times New Roman",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

# Draw tokens with colors
x = 0.01
y -= line_height
for token, reward in zip(response_tokens_1, rewards_1):
    color = cmap(norm(reward))
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color=color,
        ha="left",
        va="top",
        fontname="Times New Roman",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height


x = 0.01
y -= line_height * 2
for token in ["Example 3:"]:
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color="black",
        ha="left",
        va="top",
        fontname="Times New Roman",
        fontweight="bold",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

# Plot prompt tokens like response tokens (in black)
x = 0.01
y -= line_height
prompt_tokens = prompt_text_2.split()

for token in prompt_tokens:
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color="black",
        ha="left",
        va="top",
        fontname="Times New Roman",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

# Add response intro
x = 0.01
y -= line_height
for token in response_intro.split():
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color="black",
        ha="left",
        va="top",
        fontname="Times New Roman",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

# Draw tokens with colors
x = 0.01
y -= line_height
for token, reward in zip(response_tokens_2, rewards_2):
    color = cmap(norm(reward))
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color=color,
        ha="left",
        va="top",
        fontname="Times New Roman",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height


x = 0.01
y -= line_height * 2
for token in ["Example 4:"]:
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color="black",
        ha="left",
        va="top",
        fontname="Times New Roman",
        fontweight="bold",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

# Plot prompt tokens like response tokens (in black)
x = 0.01
y -= line_height
prompt_tokens = prompt_text_3.split()

for token in prompt_tokens:
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color="black",
        ha="left",
        va="top",
        fontname="Times New Roman",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

# Add response intro
x = 0.01
y -= line_height
for token in response_intro.split():
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color="black",
        ha="left",
        va="top",
        fontname="Times New Roman",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

# Draw tokens with colors
x = 0.01
y -= line_height
for token, reward in zip(response_tokens_3, rewards_3):
    color = cmap(norm(reward))
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color=color,
        ha="left",
        va="top",
        fontname="Times New Roman",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

x = 0.01
y -= line_height * 2
for token in ["Example 5:"]:
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color="black",
        ha="left",
        va="top",
        fontname="Times New Roman",
        fontweight="bold",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

# Plot prompt tokens like response tokens (in black)
x = 0.01
y -= line_height
prompt_tokens = prompt_text_4.split()

for token in prompt_tokens:
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color="black",
        ha="left",
        va="top",
        fontname="Times New Roman",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

# Add response intro
x = 0.01
y -= line_height
for token in response_intro.split():
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color="black",
        ha="left",
        va="top",
        fontname="Times New Roman",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

# Draw tokens with colors
x = 0.01
y -= line_height
for token, reward in zip(response_tokens_4, rewards_4):
    color = cmap(norm(reward))
    ax.text(
        x,
        y,
        token + " ",
        fontsize=fontsize,
        color=color,
        ha="left",
        va="top",
        fontname="Times New Roman",
    )
    x += word_size * len(token)
    if x > 0.95:
        x = 0.01
        y -= line_height

# Colorbar legend
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, orientation="horizontal", ax=ax, pad=0.02, shrink=0.5)
cbar.set_label("Reward Value", fontsize=fontsize)

plt.tight_layout()
plt.savefig("reward_example.pdf", dpi=300, bbox_inches="tight")
