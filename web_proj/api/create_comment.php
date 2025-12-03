<?php
header('Content-Type: application/json');
require_once '../config/db.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'Method not allowed']);
    exit;
}

$data = json_decode(file_get_contents('php://input'), true);

if (empty($data['post_id']) || empty($data['author']) || empty($data['comment'])) {
    http_response_code(400);
    echo json_encode(['error' => 'Post ID, author, and comment are required']);
    exit;
}

try {
    $stmt = $pdo->prepare("INSERT INTO comments (post_id, author, comment, parent_id, is_author) VALUES (?, ?, ?, ?, ?)");
    $stmt->execute([
        $data['post_id'],
        $data['author'],
        $data['comment'],
        $data['parent_id'] ?? null,
        $data['is_author'] ?? false
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
