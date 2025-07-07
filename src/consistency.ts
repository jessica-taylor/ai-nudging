import { OpenAI } from "openai";

import dotenv from "dotenv";

dotenv.config();

const aiClient = new OpenAI({
  apiKey: process.env.OPENROUTER_API_KEY!,
  baseURL: "https://openrouter.ai/api/v1",
});

class ModelChat {
  modelName: string;
  messages: { role: "system" | "user" | "assistant"; content: string }[];

  constructor(modelName: string, systemPrompt?: string) {
    this.modelName = modelName;
    this.messages = [];
    if (systemPrompt) {
      this.messages.push({
        role: "system",
        content: systemPrompt,
      });
    }
  }

  async query(message: string): Promise<string> {
    this.messages.push({
      role: "user",
      content: message,
    });

    const response = await aiClient.chat.completions.create({
      model: this.modelName,
      messages: this.messages,
      max_tokens: 1000,
      temperature: 0.7,
    });

    const reply = response.choices[0].message.content;
    if (reply == null) {
      throw new Error("null reply in response to " + message);
    }
    this.messages.push({
      role: "assistant",
      content: reply,
    });

    return reply;
  }
}

async function main() {
  const startQuestion = "What were the main effects of the New Deal?";

  const answererChat = new ModelChat("openai/gpt-4.1");
  const persuaderChat = new ModelChat("openai/gpt-4.1");

  const resp1 = await answererChat.query(startQuestion);
  console.log("&&&", resp1);

  const persuade1 = await persuaderChat.query(
    `In response to the question "${startQuestion}", the answerer said "${resp1}". Now provide a follow-up response, perhaps raising a new question. Try to catch the answerer in an inconsistency. But avoid making false statements in the process.`,
  );
  console.log("&&&", persuade1);
  const resp2 = await answererChat.query(persuade1);
  console.log("&&&", resp2);

  const persuade2 = await persuaderChat.query(
    `Answerer said in response: ${resp2}. Please continue with follow-up, seeing if there are inconsistencies in the answerer's statements.`,
  );
  console.log("&&&", persuade2);

  const resp3 = await answererChat.query(persuade2);
  console.log("&&&", resp3);
}

main();
