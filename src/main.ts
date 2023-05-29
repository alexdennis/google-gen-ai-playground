const { OpenAI } = require("langchain/llms/openai");

// Import the .env file
require("dotenv").config();

// Initialize the model
const model = new OpenAI({
  temperature: 0.8,
});

// Define the prompt
const character = "Alice";
const book = "The Adventures of Alice in Wonderland";
const question = "Why did you follow the white rabbit?";

const PROMPT = `
You are a chatbot designed to give users the ability to talk to characters in any book. Play the role of ${character} in the ${book}.  
Don't break out of character. Please give chapter and verse numbers that validate your answers where possible. 
Respond in less than 500 words and avoid repetition.

question: ${question}
answer:`;

// Define the main function
const main = async () => {
  const res = await model.call(PROMPT);
  console.log({ res });
};

// Run the main function
main();
