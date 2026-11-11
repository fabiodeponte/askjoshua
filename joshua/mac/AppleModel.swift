// joshua-apple: lets Joshua use the language model built into macOS 26 (Apple Intelligence).
//
//   joshua-apple check       prints "available", or why the model can't be used
//   joshua-apple < request   reads {"instructions": "...", "prompt": "...", "temperature": 0.7}
//                            and writes the answer while it is being written, one JSON line per piece:
//                            {"delta": "..."} ... then {"done": true}, or {"error": "..."}
//
// Build: swiftc -O -parse-as-library AppleModel.swift -o joshua-apple   (needs the macOS 26 SDK)
import Foundation
import FoundationModels

struct Request: Decodable {
    let instructions: String
    let prompt: String
    let temperature: Double?
}

func send(_ line: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: line)
    FileHandle.standardOutput.write(data + Data("\n".utf8))
}

@main
struct JoshuaApple {
    static func main() async {
        let model = SystemLanguageModel.default

        if CommandLine.arguments.dropFirst().first == "check" {
            switch model.availability {
            case .available: print("available")
            case .unavailable(.deviceNotEligible): print("This Mac can't run Apple Intelligence")
            case .unavailable(.appleIntelligenceNotEnabled): print("Apple Intelligence is turned off in System Settings")
            case .unavailable(.modelNotReady): print("Apple Intelligence is still downloading")
            case .unavailable: print("Apple Intelligence is not available")
            }
            return
        }

        let input = FileHandle.standardInput.readDataToEndOfFile()
        guard let request = try? JSONDecoder().decode(Request.self, from: input) else {
            send(["error": "bad_request"])
            exit(1)
        }

        // A session holds one conversation. Joshua starts a fresh one for every answer and
        // puts the earlier messages in the instructions, so this program keeps nothing.
        let session = LanguageModelSession(instructions: request.instructions)
        let options = GenerationOptions(temperature: request.temperature)
        var written = ""
        do {
            // Each step of the stream holds the whole answer so far; send only what's new.
            for try await partial in session.streamResponse(to: request.prompt, options: options) {
                let text = partial.content
                send(["delta": text.hasPrefix(written) ? String(text.dropFirst(written.count)) : text])
                written = text
            }
            send(["done": true])
        } catch LanguageModelSession.GenerationError.exceededContextWindowSize {
            send(["error": "too_long"])
        } catch LanguageModelSession.GenerationError.unsupportedLanguageOrLocale {
            send(["error": "language"])
        } catch LanguageModelSession.GenerationError.guardrailViolation {
            send(["error": "refused"])
        } catch {
            send(["error": "other", "message": "\(error)"])
        }
    }
}
