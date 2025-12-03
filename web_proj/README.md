# Simple Blog

A minimal blog system built with HTML, CSS, JavaScript, PHP, and MySQL.

## What You Need

- XAMPP (for Apache and MySQL)
- A web browser

## Setup

### 1. Start XAMPP

Open XAMPP Control Panel and start:
- Apache
- MySQL

### 2. Create the Database

Go to http://localhost/phpmyadmin

Click the SQL tab and run the contents of `database.sql`

Or just copy/paste this:

```sql
CREATE DATABASE IF NOT EXISTS simple_blog;
USE simple_blog;

CREATE TABLE IF NOT EXISTS posts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    author VARCHAR(100) NOT NULL,
    image VARCHAR(255),
    like_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS comments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    post_id INT NOT NULL,
    author VARCHAR(100) NOT NULL,
    comment TEXT NOT NULL,
    parent_id INT DEFAULT NULL,
    is_author BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_id) REFERENCES comments(id) ON DELETE CASCADE
);
```

### 3. Access the Site

Open your browser and go to:

http://localhost/vachan_proj/web_proj/

## Usage

**Create a Post**

Click "New Post" in the navigation, fill out the form, and hit publish.

**Like a Post**

Click the heart button on any post page.

**Comment**

Scroll down on any post and add your comment.

**Reply to Comments**

Click the "Reply" button under any comment. If you use the same name as the post author, you'll get an "Author" badge.

## Files

- `index.html` - Homepage with post list
- `new-post.html` - Create new posts
- `post.html` - View individual posts with comments
- `css/style.css` - All the styling
- `config/db.php` - Database connection settings
- `api/` - Backend PHP files for handling data

## Database Config

If your MySQL has a password or different username, edit `config/db.php`:

```php
$host = 'localhost';
$dbname = 'simple_blog';
$username = 'root';
$password = ''; // Add your password here if needed
```

## Troubleshooting

**Can't connect to database?**
- Make sure MySQL is running in XAMPP
- Check your credentials in `config/db.php`

**Pages not loading?**
- Verify Apache is running
- Check that files are in the correct folder: `c:\xampp\htdocs\vachan_proj\web_proj\`

**Replies not working?**
- Run `update_database.sql` if you created the database before the reply feature was added
