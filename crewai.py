"""A minimal sequential CrewAI example."""

import os

from crewai import Agent, Task, Crew, Process

#os.environ["OPENAI_API_KEY"] = "\your-api-key"  # Replace with your actual API key

researcher = Agent(
	role="Technology Researcher",
	goal="Find reliable, concise information about {topic}",
	backstory="You are a careful researcher who distinguishes facts from opinions.",
	llm_model="gpt-4o-mini",
	verbose=True,
)

writer = Agent(
	role="Technical Writer",
	goal="Turn research about {topic} into a clear short brief",
	backstory="You write accurate, readable briefs for a general audience.",
	llm_model="gpt-4o-mini",
	verbose=True,
)

research_task = Task(
	description="Research {topic}. Summarize the key points and note any uncertainty.",
	expected_output="A concise, factual list of key findings about {topic}.",
	agent=researcher,
)

writing_task = Task(
	description="Use the research findings to write a brief about {topic}.",
	expected_output="A short brief with a title, an overview, and practical takeaways.",
	agent=writer,
	context=[research_task],
)

crew = Crew(
	agents=[researcher, writer],
	tasks=[research_task, writing_task],
	process=Process.sequential,
	verbose=True,
)


if __name__ == "__main__":
	result = crew.kickoff(inputs={"topic": "how AI agents can help small businesses"})
	print(result)
