"use client"

import type { BundledLanguage } from '@/components/ui/shadcn-io/code-block';
import {
    CodeBlock,
    CodeBlockBody,
    CodeBlockContent,
    CodeBlockItem,
} from '@/components/ui/shadcn-io/code-block';

export default function JsonViewer({ json }: { json: string }) {
    const code = [
        {
            language: "json",
            filename: "input.json",
            code: json,
        }
    ]

    return (
        <CodeBlock className="max-h-full overflow-y-auto" data={code} defaultValue={code[0].language}>
            <CodeBlockBody>
                {(item) => (
                    <CodeBlockItem key={item.language} value={item.language}>
                        <CodeBlockContent
                            language={item.language as BundledLanguage}
                            themes={{
                                light: 'vitesse-light',
                                dark: 'vitesse-dark',
                            }}>
                            {item.code}
                        </CodeBlockContent>
                    </CodeBlockItem>
                )}
            </CodeBlockBody>
        </CodeBlock>
    )
}