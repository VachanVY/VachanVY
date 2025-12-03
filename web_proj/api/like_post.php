<?php
header('Content-Type: application/json');
require_once '../config/db.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'Method not allowed']);
    exit;
}

$data = json_decode(file_get_contents('php://input'), true);

if (empty($data['id'])) {
    http_response_code(400);
    echo json_encode(['error' => 'Post ID required']);
    exit;
}

try {
    $stmt = $pdo->prepare("UPDATE posts SET like_count = like_count + 1 WHERE id = ?");
    $stmt->execute([$data['id']]);
    
    $stmt = $pdo->prepare("SELECT like_count FROM posts WHERE id = ?");
    $stmt->execute([$data['id']]);
    $result = $stmt->fetch();
    
    echo json_encode([
        'success' => true,
        'like_count' => $result['like_count']
    ]);
} catch(PDOException $e) {
    http_response_code(500);
    echo json_encode(['error' => $e->getMessage()]);
}
?>
