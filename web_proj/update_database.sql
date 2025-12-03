-- Run this to update your existing database with reply functionality
-- If you get errors about columns already existing, that's fine - it means they're already there!

USE simple_blog;

-- Add new columns to comments table
ALTER TABLE comments 
ADD COLUMN parent_id INT DEFAULT NULL AFTER comment;

ALTER TABLE comments 
ADD COLUMN is_author BOOLEAN DEFAULT FALSE AFTER parent_id;

-- Add foreign key constraint for parent_id
ALTER TABLE comments 
ADD CONSTRAINT fk_parent_comment 
FOREIGN KEY (parent_id) REFERENCES comments(id) ON DELETE CASCADE;
