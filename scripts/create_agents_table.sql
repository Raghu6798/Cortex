-- Create agents table
CREATE TABLE IF NOT EXISTS agents (
    "AgentId" VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    description TEXT,
    architecture VARCHAR NOT NULL,
    framework VARCHAR NOT NULL,
    settings JSON NOT NULL,
    tools JSON,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS ix_agents_AgentId ON agents ("AgentId");
CREATE INDEX IF NOT EXISTS ix_agents_user_id ON agents (user_id);

-- Add agent_id column to chat_sessions if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'chat_sessions' AND column_name = 'agent_id') THEN
        ALTER TABLE chat_sessions ADD COLUMN agent_id VARCHAR;
    END IF;
END $$;

-- Add foreign key constraint if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints 
                   WHERE constraint_name LIKE '%agent_id%' 
                   AND table_name = 'chat_sessions') THEN
        ALTER TABLE chat_sessions ADD FOREIGN KEY (agent_id) REFERENCES agents ("AgentId");
    END IF;
END $$;
