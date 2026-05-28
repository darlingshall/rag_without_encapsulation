# main.py
from config import MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY
from agent.rag_agent import system_prompt_generator
from models.llm import create_llm
import sys
def main():

    tokenizer, model = create_llm()
    print("\n🤖  Model & Tokenizer is ready! Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_input = input("👤 You: ").strip()
            if not user_input or user_input.lower() in {"exit", "quit"}:
                print("👋 Bye!")
                break

            # 用 system_prompt_generator 根据用户提示词生成系统提示词（自动完成检索+拼接系统提示词）
            system_prompt = system_prompt_generator(user_input)

            # 构造含 system role 的消息列表
            messages = [
                {"role": "system", "content": system_prompt},  # 注意：这里 content 是纯文本！
                {"role": "user", "content": user_input}
            ]
            # 使用 tokenizer 的 apply_chat_template 方法，将对话消息列表 (messages) 格式化为模型可接受的字符串
            # tokenize=False 表示只返回格式化后的字符串，不进行 tokenization
            # add_generation_prompt=True 会在末尾添加一个特殊的提示（如 <|im_start|>assistant），告诉模型开始生成回复

            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # 将格式化后的 prompt 字符串转换为模型所需的 token ID 张量（PyTorch tensor）
            # return_tensors="pt" 指定返回 PyTorch 张量

            # 调用模型的 generate 方法生成回复
            # **inputs 将 tokenized 的输入（包括 input_ids 和 attention_mask）解包传入
            inputs = tokenizer(formatted_prompt, return_tensors="pt")

            # inputs = inputs.input_ids

            outputs = model.generate(**inputs,
                                    max_new_tokens=MAX_NEW_TOKENS,          # 限制生成的最大新 token 数量，防止无限生成
                                    temperature=TEMPERATURE,                # 控制生成的随机性：值越低越确定，越高越随机（通常 0.1-0.7），本系统若超设置为0.8则模型会再现幻觉。
                                    top_p=TOP_P,                            # 核采样 (nucleus sampling) 参数，动态选择概率累积和超过 p 的最小 token 集合
                                    repetition_penalty=REPETITION_PENALTY,  # 重复惩罚系数，>1.0 可减少重复内容（如 1.1~1.2）
                                    do_sample=True,                         # 启用基于采样的生成（而非贪心搜索），与 temperature/top_p 配合使用
                                    eos_token_id=tokenizer.eos_token_id,    # 指定“结束生成”的 token ID（如 </s>），模型遇到它会停止
                                    # 填充 token ID，若未定义则用 eos_token_id 代替
                                    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                                     )

            # 获取输入 prompt 的长度（token 数量），用于后续从完整输出中截取仅模型生成的部分
            input_length = inputs["input_ids"].shape[1]

            # 从模型的完整输出中，跳过输入部分，只解码模型生成的新 token
            # skip_special_tokens=True 会移除特殊 token（如 <s>, </s>, <|user|> 等），使输出更干净
            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)


            print(f"🤖 AI: {response}\n")
            # messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("👋 Bye!")
            sys.exit(0)

if __name__ == "__main__":
    main()