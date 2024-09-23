import json

def handleMarketEvent():
    """Handles the 'Local market day' event."""
    global player_score
    print("\n[!] You've stumbled upon the lively local market! The vibe is buzzing!")

    while True:
        choice = input("\n[?] Wanna check it out? (yes/no): ")
        if choice.lower() == 'yes':
            print("\n[👀] You stroll through the stalls, the aroma of fresh patties fills the air.")
            print("\n[🎁] You grab a tasty snack from a food stall.")
            player_inventory.append("Market Snack") 
            player_score += 1 
            break 
        elif choice.lower() == 'no':
            print("\n[🏃‍♂️] Aight, back on the bus then.")
            break
        else:
            print("\n[😕] Not sure what you mean, fam. Try 'yes' or 'no'.")

def handleStreetPerformerEvent():
    """Handles the 'Encounter a street performer' event."""
    global player_score
    print("\n[!] A sick beatboxer is spitting fire on the corner!")

    while True:
        choice = input("\n[?] You stopping to watch? (yes/no): ")
        if choice.lower() == 'yes':
            print("\n[🎤] You vibe to the rhythm, throwing a quid in the hat.")
            player_score += 1 
            break
        elif choice.lower() == 'no':
            print("\n[🚶‍♂️] You keep it moving, the beat fading as you walk away.")
            break
        else:
            print("\n[😕] Not sure what you mean, fam. Try 'yes' or 'no'.")

with open("game_data.json", "r") as f:
    game_data = json.load(f)

game_title = game_data["settings"]["title"]
print(f"Welcome to {game_title}, let’s roll!")

current_location = game_data["settings"]["startLocation"]
current_stop_index = 0

player_score = 0
player_inventory = []

while current_location != game_data["settings"]["endLocation"]:
    print(f"\n[🚌] You roll up to {current_location}, the bassline thumping in your ears.")

    for event in game_data["settings"]["events"]:
        if event["location"] == current_location:
            print(f"[!] {event['description']}")

            if "handler" in event:
                handler_function = globals().get(event["handler"])
                if handler_function:
                    handler_function()
            elif "options" in event:
                for i, option in enumerate(event["options"]):
                    print(f"{i + 1}. {option}")
                player_choice = input("Choose an option: ")
                if player_choice.isdigit() and 1 <= int(player_choice) <= len(event["options"]):
                    print(f"You chose: {event['options'][int(player_choice) - 1]}")
                else:
                    print("Invalid choice, mate.")

    player_choice = input("[?] Wagwan? What you gonna do next? (Type 'next' to blaze to the next stop) ")

    if player_choice.lower() == 'next':
        current_stop_index += 1
        if current_stop_index < len(game_data["settings"]["stops"]): 
            current_location = game_data["settings"]["stops"][current_stop_index]["name"]
        else:
            current_location = game_data["settings"]["endLocation"] 
    else:
        print("Invalid action. Type 'next' to keep moving.")

print(f"\n[🏁] Manchester, innit! You’ve hit the end of the line with a score of {player_score}. Safe travels, fam!")
