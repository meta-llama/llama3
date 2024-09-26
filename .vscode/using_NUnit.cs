using NUnit.Framework;

public class EventTriggerTests
{
    private Player player;
    private Location alkrington;
    
    [SetUp]
    public void Setup()
    {
        player = new Player();
        alkrington = new Location("Alkrington");
        alkrington.AddEvent("Traffic Jam");
    }
    
    [Test]
    public void TestTrafficJamEventTriggered()
    {
        // Simulate the player reaching Alkrington
        player.MoveTo(alkrington);
        
        // Check if the event is triggered when the player arrives
        Assert.IsTrue(alkrington.IsEventTriggered("Traffic Jam"));
    }
}
