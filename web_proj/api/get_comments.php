<?php
header('Content-Type: application/json');
require_once '../config/db.php';

if (!isset($_GET['post_id'])) {
    http_response_code(400);
    echo json_encode(['error' => 'Post ID required']);
    exit;
}

try {
    $stmt = $pdo->prepare("SELECT * FROM comments WHERE post_id = ? ORDER BY created_at ASC");
    $stmt->execute([$_GET['post_id']]);
    $comments = $stmt->fetchAll();
    echo json_encode($comments);
} catch(PDOException $e) {
    http_response_code(500);
    echo json_encode(['error' => $e->getMessage()]);
}
?>
