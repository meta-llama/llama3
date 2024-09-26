// Example location class
public class Location
{
    public string Name { get; set; }
    public Dictionary<string, bool> Events { get; set; }

    public bool IsEventTriggered(string eventName)
    {
        return Events.ContainsKey(eventName) && Events[eventName];
    }

    // Add more logic here as needed for events, conditions, etc.
}

// Example for Langley location
var langley = new Location 
{
    Name = "Langley",
    Events = new Dictionary<string, bool>
    {
        { "First Pre-Roll Sale", true }
    }
};
