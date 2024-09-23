game_data = {
    "title": "©️18BluntWrapz",
    "locations": [
        {"name": "Langley", "events": ["Local Market Day"]},
        {"name": "Middleton Bus Station", "events": ["Street Performer Encounter"]},
        {"name": "Alkrington", "events": ["Traffic Jam Surprise"]},
        {"name": "Blackley", "events": ["Missed Stop"]},
        {"name": "Harpurhey", "events": ["Bus Breakdown Challenge"]},
        {"name": "Collyhurst", "events": ["Hidden Gem Discovery"]},
        {"name": "North Manchester General Hospital", "events": ["Community Health Fair"]},
        {"name": "Manchester Royal Infirmary", "events": ["Meet the Characters"]},
        {"name": "Manchester Piccadilly Gardens", "events": ["City Vibes"]},
        {"name": "Manchester Arndale", "events": ["Shopping Spree"]},
        {"name": "Manchester Victoria Station", "events": ["Final Destination"]},
    ],
    "events": {
        "Local Market Day": {
            "description": "Explore vibrant stalls filled with fresh produce and handmade crafts. Will you indulge?",
            "handler": "handleMarketEvent"
        },
        "Street Performer Encounter": {
            "description": "Catch a talented street performer showcasing their skills. Will you show your support?",
            "handler": "handleStreetPerformerEvent"
        },
        "Traffic Jam Surprise": {
            "description": "A traffic jam halts your journey. Will you wait it out or explore on foot?",
            "handler": "handleTrafficJam"
        },
        "Missed Stop": {
            "description": "Oops! You missed your intended stop. What will you do next?",
            "handler": "handleMissedStop"
        },
        "Bus Breakdown Challenge": {
            "description": "The bus breaks down unexpectedly. Find a way to continue your adventure!",
            "handler": "handleBusBreakdown"
        },
        "Hidden Gem Discovery": {
            "description": "Uncover local spots filled with culture and charm.",
            "handler": "handleHiddenGem"
        },
        # Add more events as needed...
    }
}


