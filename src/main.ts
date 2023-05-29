const {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} = require("langchain/prompts");
const { LLMChain } = require("langchain/chains");
const { ChatOpenAI } = require("langchain/chat_models/openai");
const { program } = require("commander");

// Import the .env file
require("dotenv").config();

// Define our program
program
  .name("book-chatbot")
  .description("CLI to talk to characters in books")
  .version("alpha");

// Define the chat command
program
  .command("chat")
  .description("Chat with a character in a book")
  .argument("<book>", "Book name")
  .argument("<character>", "Book character")
  .argument("<question>", "Question to ask")
  .action(async (book: string, character: string, question: string) => {
    console.info({ book, character, question });
    const llmChain = createChain();
    const res = await llmChain.call({
      character,
      book,
      question,
    });
    console.info({ res });

    if (res.error) {
      console.error(res.error);
    } else if (res.text) {
      console.log(res.text);
    }
  });

// Parse CLI input
program.parse();

/**
 * Chreate a new LLMChain
 *
 * @param temperature Default 0 to give deterministic results
 * @returns
 */
function createChain(temperature: number = 0) {
  const chat = new ChatOpenAI({ temperature });
  const chatPrompt = ChatPromptTemplate.fromPromptMessages([
    SystemMessagePromptTemplate.fromTemplate(
      `You are a chatbot designed to give users the ability to talk to characters in any book. Play the role of {character} in the {book}.  
Don't break out of character. Please give chapter and verse numbers that validate your answers where possible. 
Respond in less than 500 words and avoid repetition.`
    ),
    HumanMessagePromptTemplate.fromTemplate("{question}"),
  ]);
  return new LLMChain({
    prompt: chatPrompt,
    llm: chat,
  });
}
