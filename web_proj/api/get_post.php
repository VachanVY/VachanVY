<?php
header('Content-Type: application/json');
require_once '../config/db.php';

if (!isset($_GET['id'])) {
    http_response_code(400);
    echo json_encode(['error' => 'Post ID required']);
    exit;
}

try {
    $stmt = $pdo->prepare("SELECT * FROM posts WHERE id = ?");
    $stmt->execute([$_GET['id']]);
    $post = $stmt->fetch();
    
    if ($post) {
        echo json_encode($post);
    } else {
        http_response_code(404);
        echo json_encode(['error' => 'Post not found']);
    }
} catch(PDOException $e) {
    http_response_code(500);
    echo json_encode(['error' => $e->getMessage()]);
}
?>
