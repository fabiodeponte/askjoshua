<?php
// Receives the beta-test form (feedback.html) and saves each answer as one line of JSON,
// in a folder outside the website. No address, no cookie: only what the person wrote, and the date.
$DIR = '/var/www/askjoshua-feedback';
$LANGS = ['en', 'it', 'fr', 'de', 'pl', 'zh'];

$lang = $_POST['lang'] ?? 'en';
$lang = in_array($lang, $LANGS, true) ? $lang : 'en';
$back = ($lang === 'en' ? '/' : "/$lang/") . 'feedback.html';

if ($_SERVER['REQUEST_METHOD'] !== 'POST' || !empty($_POST['website'])) {  // "website" is a hidden field only robots fill in
    header("Location: $back"); exit;
}
$text = fn($k, $max) => mb_substr(trim((string)($_POST[$k] ?? '')), 0, $max);
$tasks = [];
foreach ((array)($_POST['task'] ?? []) as $name => $result) {
    if (preg_match('/^[a-z]{2,12}$/', $name) && in_array($result, ['worked', 'partly', 'failed'], true)) {
        $tasks[$name] = $result;
    }
}
$entry = [
    'date' => gmdate('Y-m-d H:i'),
    'lang' => $lang,
    'version' => $text('version', 20),
    'computer' => $text('computer', 120),
    'model' => $text('model', 120),
    'before' => $text('before', 120),
    'tasks' => $tasks,
    'replace' => in_array($_POST['replace'] ?? '', ['yes', 'some', 'no'], true) ? $_POST['replace'] : '',
    'comment' => $text('comment', 4000),
    'email' => filter_var($text('email', 200), FILTER_VALIDATE_EMAIL) ?: '',
];
$saved = @file_put_contents("$DIR/feedback.jsonl", json_encode($entry, JSON_UNESCAPED_UNICODE) . "\n", FILE_APPEND | LOCK_EX);
header("Location: $back?" . ($saved ? 'thanks' : 'error'));
