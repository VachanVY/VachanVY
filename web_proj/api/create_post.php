<?php
header('Content-Type: application/json');
require_once '../config/db.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'Method not allowed']);
    exit;
}

$data = json_decode(file_get_contents('php://input'), true);

if (empty($data['title']) || empty($data['content']) || empty($data['author'])) {
    http_response_code(400);
    echo json_encode(['error' => 'Title, content, and author are required']);
    exit;
}

try {
    $stmt = $pdo->prepare("INSERT INTO posts (title, content, author, image) VALUES (?, ?, ?, ?)");
    $stmt->execute([
        $data['title'],
        $data['content'],
        $data['author'],
        $data['image'] ?? null
    ]);
    
    echo json_encode([
        'success' => true,
        'id' => $pdo->lastInsertId()
    ]);
} catch(PDOException $e) {
    http_response_code(500);
    echo json_encode(['error' => $e->getMessage()]);
}
?>
