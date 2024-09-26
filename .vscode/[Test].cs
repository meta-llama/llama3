[Test]
public void TestTokenizerHandlesBasicInput()
{
    var tokenizer = new Tokenizer();
    var tokens = tokenizer.Tokenize("Smoke weed everyday");
    
    Assert.AreEqual(3, tokens.Count);
    Assert.AreEqual("Smoke", tokens[0]);
    Assert.AreEqual("weed", tokens[1]);
    Assert.AreEqual("everyday", tokens[2]);
}

[Test]
public void TestTokenizerHandlesEmptyInput()
{
    var tokenizer = new Tokenizer();
    var tokens = tokenizer.Tokenize("");
    
    Assert.AreEqual(0, tokens.Count); // Empty input should result in no tokens
}

[Test]
public void TestTokenizerHandlesSpecialCharacters()
{
    var tokenizer = new Tokenizer();
    var tokens = tokenizer.Tokenize("Cannabis & Blunts!!");

    Assert.AreEqual(3, tokens.Count);
    Assert.AreEqual("Cannabis", tokens[0]);
    Assert.AreEqual("&", tokens[1]);
    Assert.AreEqual("Blunts", tokens[2]);
}
